"""Reset the MM dashboard login (Fly tarakta-mm2, optionally local .env).

Run interactively:  python3 scripts/reset_dashboard_login.py
Prompts for a new username/password (hidden), bcrypt-hashes locally, sets
the Fly secrets (machine auto-restarts), and can align the local bot too.
The plaintext password never leaves this process.
"""
import getpass
import subprocess

import bcrypt

APP = "tarakta-mm2"
ENV_PATH = "/Users/stuarta/tarakta/.env"


def main() -> None:
    user = input("New dashboard username: ").strip()
    if not user:
        raise SystemExit("Empty username — nothing changed.")
    pw = getpass.getpass("New password (hidden): ")
    if pw != getpass.getpass("Confirm password: "):
        raise SystemExit("Passwords did not match — nothing changed.")
    if len(pw) < 8:
        raise SystemExit("Use at least 8 characters — nothing changed.")

    hashed = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    subprocess.run(
        ["fly", "secrets", "set", "--app", APP,
         f"DASHBOARD_USERNAME={user}", f"DASHBOARD_PASSWORD_HASH={hashed}"],
        check=True,
    )
    print(f"Done — {APP} is restarting with the new login.")

    if input("Also update the LOCAL bot's login in .env? [y/N]: ").lower() == "y":
        lines = open(ENV_PATH).read().splitlines()
        keep = [
            l for l in lines
            if not l.startswith(("DASHBOARD_USERNAME=", "DASHBOARD_PASSWORD_HASH="))
        ]
        keep += [f"DASHBOARD_USERNAME={user}", f"DASHBOARD_PASSWORD_HASH={hashed}"]
        open(ENV_PATH, "w").write("\n".join(keep) + "\n")
        subprocess.run(
            ["launchctl", "kickstart", "-k", "gui/501/com.tarakta.bot"],
            check=True,
        )
        print("Local .env updated and local bot restarted — same login everywhere.")


if __name__ == "__main__":
    main()
