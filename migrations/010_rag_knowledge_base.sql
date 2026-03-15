-- RAG Knowledge Base for Trade Intelligence
-- Ported from doubleclicker hybrid RAG system
-- Enables Agent 1 and Agent 2 to learn from historical trade outcomes

-- Enable pgvector extension (idempotent)
create extension if not exists vector;

-- ============================================================================
-- Table: knowledge_sources
-- One row per ingested document/trade
-- ============================================================================

create table if not exists knowledge_sources (
  id uuid primary key default gen_random_uuid(),
  user_name text not null,
  domain text not null,
  source_type text not null,
  source_url text not null,
  title text,
  description text,
  status text not null default 'queued',
  content_hash text,
  last_ingested_at timestamptz,
  error text,
  metafields jsonb not null default '{}'::jsonb,
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  unique (user_name, domain, source_type, source_url)
);

create index if not exists idx_knowledge_sources_user_domain
  on knowledge_sources(user_name, domain);
create index if not exists idx_knowledge_sources_type
  on knowledge_sources(source_type);
create index if not exists idx_knowledge_sources_status
  on knowledge_sources(status);
create index if not exists idx_knowledge_sources_hash
  on knowledge_sources(content_hash);

-- ============================================================================
-- Table: knowledge_chunks
-- Chunked content with embeddings (1536d OpenAI text-embedding-3-small)
-- ============================================================================

create table if not exists knowledge_chunks (
  id uuid primary key default gen_random_uuid(),
  user_name text not null,
  domain text not null,
  source_id uuid not null references knowledge_sources(id) on delete cascade,
  chunk_index int not null,
  chunk_hash text not null,
  content text not null,
  content_tsv tsvector generated always as (
    to_tsvector('english', coalesce(content, ''))
  ) stored,
  embedding vector(1536),
  url text,
  heading text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  unique (source_id, chunk_index)
);

create index if not exists idx_knowledge_chunks_user_domain
  on knowledge_chunks(user_name, domain);
create index if not exists idx_knowledge_chunks_source
  on knowledge_chunks(source_id);
create index if not exists idx_knowledge_chunks_tsv
  on knowledge_chunks using gin(content_tsv);
create index if not exists idx_knowledge_chunks_embedding
  on knowledge_chunks using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- ============================================================================
-- RPC: Dense semantic search
-- ============================================================================

create or replace function rag_dense_search(
  p_user_name text,
  p_domain text,
  p_query_embedding vector(1536),
  p_k int default 50,
  p_min_similarity float default 0.15
)
returns table (chunk_id uuid, rank bigint, similarity float)
language sql stable as $$
  select
    id as chunk_id,
    row_number() over (order by embedding <=> p_query_embedding) as rank,
    (1 - (embedding <=> p_query_embedding))::float as similarity
  from knowledge_chunks
  where user_name = p_user_name
    and domain = p_domain
    and embedding is not null
    and (1 - (embedding <=> p_query_embedding)) >= p_min_similarity
  order by embedding <=> p_query_embedding
  limit p_k;
$$;

-- ============================================================================
-- RPC: Lexical keyword search
-- ============================================================================

create or replace function rag_lexical_search(
  p_user_name text,
  p_domain text,
  p_query text,
  p_k int default 50
)
returns table (chunk_id uuid, rank bigint, score float)
language sql stable as $$
  select
    kc.id as chunk_id,
    row_number() over (order by ts_rank_cd(kc.content_tsv, q.tsq) desc) as rank,
    ts_rank_cd(kc.content_tsv, q.tsq)::float as score
  from knowledge_chunks kc, (
    select websearch_to_tsquery('english', p_query) as tsq
  ) q
  where kc.user_name = p_user_name
    and kc.domain = p_domain
    and kc.content_tsv @@ q.tsq
  order by score desc
  limit p_k;
$$;

-- ============================================================================
-- RPC: Hybrid search with Reciprocal Rank Fusion (RRF)
-- ============================================================================

create or replace function rag_hybrid_search(
  p_user_name text,
  p_domain text,
  p_query text,
  p_query_embedding vector(1536),
  p_k int default 50,
  p_rrf_k int default 60,
  p_min_similarity float default 0.15
)
returns table (
  chunk_id uuid,
  content text,
  url text,
  heading text,
  metadata jsonb,
  rrf_score float,
  dense_rank bigint,
  lexical_rank bigint,
  similarity float
)
language sql stable as $$
  with dense_results as (
    select * from rag_dense_search(p_user_name, p_domain, p_query_embedding, p_k, p_min_similarity)
  ),
  lexical_results as (
    select * from rag_lexical_search(p_user_name, p_domain, p_query, p_k)
  ),
  combined as (
    select
      coalesce(d.chunk_id, l.chunk_id) as chunk_id,
      d.rank as dense_rank,
      l.rank as lexical_rank,
      d.similarity,
      coalesce(1.0 / (p_rrf_k + d.rank), 0.0) +
      coalesce(1.0 / (p_rrf_k + l.rank), 0.0) as rrf_score
    from dense_results d
    full outer join lexical_results l on d.chunk_id = l.chunk_id
  )
  select
    c.chunk_id,
    kc.content,
    kc.url,
    kc.heading,
    kc.metadata,
    c.rrf_score,
    c.dense_rank,
    c.lexical_rank,
    c.similarity
  from combined c
  join knowledge_chunks kc on kc.id = c.chunk_id
  order by c.rrf_score desc
  limit p_k;
$$;

-- ============================================================================
-- Updated_at triggers
-- ============================================================================

create or replace function update_updated_at_column()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

-- Only create triggers if they don't exist (avoid errors on re-run)
do $$
begin
  if not exists (select 1 from pg_trigger where tgname = 'update_knowledge_sources_updated_at') then
    create trigger update_knowledge_sources_updated_at
      before update on knowledge_sources
      for each row execute function update_updated_at_column();
  end if;
  if not exists (select 1 from pg_trigger where tgname = 'update_knowledge_chunks_updated_at') then
    create trigger update_knowledge_chunks_updated_at
      before update on knowledge_chunks
      for each row execute function update_updated_at_column();
  end if;
end;
$$;

-- ============================================================================
-- RLS — service role bypass
-- ============================================================================

alter table knowledge_sources enable row level security;
alter table knowledge_chunks enable row level security;

-- Drop and recreate policies (idempotent)
drop policy if exists "Service role full access knowledge_sources" on knowledge_sources;
create policy "Service role full access knowledge_sources"
  on knowledge_sources using (true) with check (true);

drop policy if exists "Service role full access knowledge_chunks" on knowledge_chunks;
create policy "Service role full access knowledge_chunks"
  on knowledge_chunks using (true) with check (true);
