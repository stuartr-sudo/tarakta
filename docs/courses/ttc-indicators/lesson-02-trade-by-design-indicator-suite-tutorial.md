# Lesson 02: The Trade By Design Indicator Suite Tutorial

**Course:** Indicator and Liquidity Software Tutorials
**Source:** https://tradetravelchill.club/lessons/the-trade-by-design-indicator-tutorial/
**Video:** Youtube `1RcKfoK53EY`

## Materials

Open Interest Indicator - click here
PVSRA Indicator - click here
Market Session Times Indicator - click here
Add to favourites within TradingView and then add them to your chart.

For access to the TBD Indicator, First Class and Capital Club Members will find this in "invite only scripts" within your tradingview account AFTER entering your TradingView username in your TTC website account settings.

- [click here](https://www.tradingview.com/script/q8rWy6C3-Open-Interest-Auto-Override/)
- [click here](https://www.tradingview.com/script/idVi84zW-PVSRA-Candles-Auto-Override/)
- [click here](https://www.tradingview.com/script/zTwIA4Ni-Market-Session-Times/)

## Transcript

[00:00] right we're going to go through the new trade by Design indicator Suite so that's not one that's not two that is three new indicators for you so going
[00:10] from an existing chart layout that you might have where you've got the trade by Design indicator what you want to do is press the X to remove it and then you
[00:20] want to reload the page so that when you add the indicator back again it's going to pick up the new code not cach code from the indicator you had before so
[00:30] after you've done that on your keyboard go controlr if you're on Windows or command R if you're on Mac you're going to choose save and reload your chart
[00:40] will come back plain then what you want to do right click go to settings symbol body borders and Wick you want those checked on
[00:50] historically they've been off and set your candle colors however you want them as a default then we'll go and add all of the
[01:00] indicators and then we'll go through them one by one so either click on indicators or on your keyboard press forward slash invite only you want to
[01:10] choose trade by Design method that's by TTC tools then we're going to look for our
[01:20] open interest so open Interest Auto override by trade travel chill you want to look for our PVS
[01:30] also Auto override by trade trael chill and we want to add our Market
[01:40] session times right we'll just go through them
[01:50] one by one so pvsr what we've introduced is this volume override or binance charts so this means
[02:00] that you don't need to use the override box anymore you don't need a different temp template layout for Bitcoin Ean other you can just use this indicator
[02:10] out the box no need to change any settings we will go and look for the binance per chart of whatever we're doing so here I'm using a spot chart for
[02:20] Bitcoin and okx this is giving me the binance per volume for for Bitcoin if I choose a chart that doesn't
[02:30] have a binance pair then it will return the chart that I'm working on so for example
[02:40] if I go nakod games this doesn't exist on barand I've
[02:50] got volume over here so no this settings are required to
[03:00] change on here I'm just going to remove these as I go so we've got more screen to work with all right so same thing here no
[03:10] settings need to be changed you can use it as is we have leftt the option there for you if you need to for some reason which I can't think of a reason why
[03:20] you'd want to force the symbol to override it is there but if you just use it as it comes then your candles and
[03:30] you know everything that's going to match cabin crew shs so that replies to the pvsr you you won't have a candle mismatch if you have this box um selected and again this works if there's
[03:40] no binance for the chart it will show the open interest for the exchange that
[03:50] you're on if the exchange you're on doesn't provide the open interest data you're going to get a study error here it's going to be blank that's perfectly
[04:00] normal so going back to the Nakamoto games example you can see it will just show the study area here because C coin
[04:10] does not provide open interest data and nakoa games doesn't exist on binance so that is all 100%
[04:20] normal right jump back onto to almost the main event so what we
[04:30] going to do after adding um tray Bon you've got two options either you want to click
[04:40] on the indicator name three dots visual order bring to front or you want to go
[04:50] into your object tree and drag the indicator to the top so you can see as I've dragged it to the top of the object tree
[05:00] if I go back to visual order bring to front is grade out so those two things will do exactly the same while we have this open I'll just go to the data
[05:10] window just to show you here for trade by Design you can see all of the levels are being returned so all
[05:20] the levels that we have and EMAs in the chart it's showing you the values in here as well
[05:30] before we jump into details of trade byan I'm just going to swing through the market session times in case somebody is new and hasn't used any of these yet we
[05:40] do have a separate video on Market session time so I'll just make a very quick uh run through but just to be complete uh let's go for
[05:50] it so we've got Moon cycles um and the option to on a higher time frame to only show the higher uh
[06:00] only show the major moon phases otherwise you end up getting quite a mess of uh of moons everywhere so there's an example full moon the
[06:10] pictures on Mac admittedly are much prettier um right so do you want to see the session bar that's this thing over here do you want to be at the top or the
[06:20] bottom so currently it's set to bottom do we want to see the labels so that's the Asia UK US you can get rid of those if you don't need that that
[06:30] clue projects the session into the future now what this does is it goes and as you can see this is where we are now but the two future sessions or three
[06:40] future sessions are showing the only time you need to pause with this is when the clocks
[06:50] change because the logic of the way that these lines create they just use the existing um position and add on the
[07:00] correct number of sessions in the front so it doesn't take into account the clock like the actual sessions do so that's just something to keep in mind just don't use it on the day when when
[07:10] the clocks change and we all know when the clocks change because of all the other drama it causes so all
[07:20] good then we've got background shading how dark or light do you want to be just play with the number here by default we've only got the dead Gap um with the
[07:30] background shading but if you wanted um the UK gap for example to be shaded as well you would just uh check the box and
[07:40] you can make it whatever color you want some of the cabin crew have got the rainbow for their session times if you want to match that you just come in here and play with uh these settings um show
[07:50] beginning of the day and um end of the day um and week lines so that's just a DOT of
[08:00] line that will show there and it will only show Once at the end of the week if you choose the week one a higher time frame session bar what this does is it places on on like on the
[08:10] 4H hour time frame where we don't show the individual um sessions this will just show a a grade lock which just
[08:20] helps you when you're back testing to give you an idea of um where the you know where the actual weekday week days were and then you can choose to shade the the
[08:30] different days um in that week if you wanted to for example you back testing Mondays you could go and set the
[08:40] background for Mondays only another thing that is very handy and it's not used that much as the alerts on the
[08:50] market session times so how alloud work alerts works is you go and check whichever ones you want to be alerted about this is far too many for me to be alerted about so I will just go and
[09:00] uncheck the ones that I'm not interested in then you go to the three dots and you go ADD alert on Market
[09:10] sessions and then this uh condition here the any alert function call what it will do is it will fit the alert for all of those ones that that you have chosen um
[09:20] the other option of course you can just set them one at a time if you want but there's really no need to do that you can just set them all in one go
[09:30] all right we'll get on to the main event I'm just going to get rid of the
[09:40] moons and here we go all right these are the new settings the first big excitement is the back test mode so if
[09:50] you check on back test mode what will happen is our levels are
[10:00] going to print back as far as the chart can go the only thing that you need to keep in mind is that the IOD and I LOD will
[10:10] only print nine weeks back so if you want data further back than that all you need to do is click on
[10:20] replay so you can see there's no um I hod lods here once you cut it off
[10:30] we're back to December even and we've got we've got the
[10:40] levels I to turn off uh back test mode the number of days you've got to display
[10:50] it's defaulted to one um but you can change it up to 7 Days um to go back the levels as we've had before the
[11:00] low and the high of the week do we want to have a label on it so here H for example now the labels will
[11:10] only display on today's date except for the IOD and IOD um and that is because just to save you know
[11:20] there's a lot of possible lines that you can add to your chart so it's just to avoid clutter the reason why we keep the I hod and IOD there is because we've got
[11:30] the percentage that can potentially show um which is quite
[11:40] nice you can change the line colors if you want the it is a bit more muted than the current version so if you prefer the red lines for the high the week you can
[11:50] just come and change them back here another um thing that we've added I'm just going to
[12:00] add the projection back on just to help us see this option we've got this EXT
[12:10] which means extension at the end of all of our our key levels so what this will do is it will extend the line out to the end of the
[12:20] trading day so you'll see the high of the week should jump out to over here so you can do them for individual lines or you can do them all in one go so extend
[12:30] all lines override will do that you check on the box all of the lines that we have will go out into the
[12:40] future so even if you have selected one or two extend all we will'll override
[12:50] that if you only using uh number of data to display one then this from day for is available to
[13:00] you if you check it it just means that the high of the day will start printing from the from the previous day just to
[13:10] give you more of a a guideline but that's only if you are using a single day for levels we've also added these 50% levels
[13:20] so 50% of the high and low of the week and 50% of the high and low of the day
[13:30] so you can see here here's a 50% of the high and
[13:40] low it also shows us the price that it's at when you hover over
[13:50] it I just have to scroll a bit so here is the 50% high low of the week so on the initial levels the ihod
[14:00] IOD we've given you the option to change the line style um just because there's a lot of lines and they
[14:10] all uh can blend into one at times so if you wanted to have a dotted line for example you could make that a dotted
[14:20] line or an arrow for example um we've also added the 50% Asia line which will autocalculate just going to get rid of these just
[14:30] o so I've just changed my number of
[14:40] to so I've just changed my number of days back to three just so we can see the 50% AIA line which is this sort of dash line so that will autocalculate for
[14:50] you that line is from 3:00 a.m. New York and or just if you've got the extension going is going to go to the
[15:00] the nd next option we have is the US fettle
[15:10] nd next option we have is the US fettle nd next option we have is the US fettle so the US FAL is the last 15 minute candle of the US session on a Friday so if I check
[15:20] that you'll see here settle also you you can also extend it um and this is just
[15:30] highlighting this gray candle here which is the last one of the day this box is only going to display on the 5 minute
[15:40] chart and the 15minute chart it's not going to display on the hour if you want to see the level on your 1 hour chart just come in here and just draw a box
[15:50] around it yourself let's get rid of
[16:00] that so line thickness if you want your Lin to be thicker or thinner you can change them here we've already gone through extend everything show warning
[16:10] if the high and the low of the week is not displayed so on the one and two minute chart the high of the week low of the week won't be displayed so if you
[16:20] just leave this box box checked you know if you if you're a scalper you want to go and draw your line manually for the high of the week if you're going down to the the one minute chart and then it
[16:30] will show you a warning here that we're not showing the lines so I've just jumped back to the 50
[16:40] minute time frame we're going to look at the IOD IH Levels by default the lines sto moving at 2 a.m. New York as you can
[16:50] see this candle 215 has gone above that we have the option of using 1:00 2:00 or 3:00 so if I change it to 3 we going to see the line keeps moving up and that
[17:00] will only stop then at uh 3:00 a.m. New York let's leave that at the default
[17:10] we've got the option to not include the de Gap Zone when we're looking at um the levels being
[17:20] drawn and again going back to the example of a lot of clutter with a lot of data for the past um you've got the option just to hide the percentage that
[17:30] displays in line with the IH levels so if you turn that off you can still just hover over it and it will tell you the percentage so that's just a personal
[17:40] preference how many days you're you're going back by by sort of standard um display 50% Asia
[17:50] from again 3:00 is the default because Asia calculates 8:00 a.m. to at 8:00 p.m. to 300 a.m. New York
[18:00] time so again it's 2:30 now we've got the option of changing it to two so it
[18:10] will start calculating early so if you want to start sort of guesstimating oh this is going to be my approximate 50% um AIA line you can change it to two U
[18:20] it will still do the calculation the 8 to3 and that that will be the final the final level um you can just see if I change us
[18:30] to today's you can see yesterday's level here it's say dash line for 50% Asia again no label in the
[18:40] past that's normal we only have the label on today data for that not not
[18:50] cluttering things up okay next one is a big section so unrecovered vector is we recommend that you leave them off when
[19:00] you're going through all of your charts in the day if you've got you know 10 20 charts that you flick through leave it off until you actually want to use the unrecovered vectors it's a huge
[19:10] calculation that goes on in the background because we're not just showing unrecovered vectors from this time frame you can show them from a different time frame so we can use we
[19:20] can be on the 50 minute chart as they show us the vectors that exist you know on the 4 Hour we'll just start with a a chart one now and I'll just turn that
[19:30] on you can choose um how many zones you want to see above and below price I'm
[19:40] just going to leave it at one now just to make it sort of easier to see and I'll just calculate and remove the others
[19:50] so we've got this option here to say exclude vectors that span price now what this means is that so
[20:00] just by looking at this chart this red candle here is a vector it's unrecovered we haven't highlighted it and that's because of the setting so you'll see if
[20:10] I turn that off it's going to highlight that red one and this isn't really helpful to you
[20:20] um so when you I found that when you are using charts or you're on the same time frame then having this exclude is pretty
[20:30] helpful so the reason why this is jumped down is because you
[20:40] can see that tiny gray dotted line it is touching this candle and we've got this now turned on to exclude the vector at span price so this candle is part of
[20:50] where price currently is therefore don't highlight it so that's why we've got the next one um highlighted which is this big um red
[21:00] one then there's the option to include Wix when you mark the vectors that's pretty obvious like on this candle we'll
[21:10] jump up here and get rid of the in bit this one here we'll get rid of the top bit as a standard um the ones the
[21:20] settings that are default is what any uses so if you want to have your CH any this is how her aled to um the other option is do you want to
[21:30] include Wix when we reclaim vectors or not now you can see I've unchecked that
[21:40] nothing has changed on the chart that's going to work with this setting to only show the unrecovered portion so as soon as I check that we should jump down to
[21:50] here um if I check it
[22:00] so you can see when I've got that include the work when reclaiming vectors and only show the the unrecovered
[22:10] portion that is how it will display you can choose to extend the vectors to go further out and change the
[22:20] colors uh the 50% line uh you can choose to not have it at all but that's 50% um of the candle and
[22:30] if for example this Wick had come down a little further and and we're saying that uh Wicks do recover um candles
[22:40] then that 50% line would disappear because we've got this hard if recovered if we don't have this set but
[22:50] the but the candle is getting recovered you can have sort of a floating 50% line out out of the box so I just keep that
[23:00] on then what we have is um an invalidation percentage so if you don't want um you know you sometimes get like
[23:10] a tiny sliver of a of a candle and in your mind you consider that to be recovered you can play with this um
[23:20] setting to see what percentage works for you so 97 998 99 um op probably the most applicable values to show you an
[23:30] example um I'm going to make this dramatic and I'm going to say inv validation percentage is five hopefully that's 5% that's been recovered yeah so
[23:40] um that will then move on to the next um unrecovered Zone because I'm saying I only want 5% to be recovered again
[23:50] dramatic example just that you can see it happening but play with it yourself see what you're happy with the default is not 99 but 97 98 99 um normally
[24:00] normally works so before we jump to the next section uh the next uh setting I'm just going to choose a higher time frame
[24:10] so what we'll do is we'll choose 4our and vectors I'm going to go to the 4our
[24:20] Chart so you can see here we've got the 4our area of Interest showing and if we go back down let's go to the 1
[24:30] hour and again it's going to take a while to calculate so just be patient
[24:40] and then you can see here 4our area of interest um and it's got you know half of this candle highlighted
[24:50] um you know it's Jo these two together so you can see this isn't a 1H hour Vector it's coming from a different time frame just to make it more obvious and we go down to 15 minute
[25:00] um let's go
[25:10] five so you can see here you've got like a red candle in the middle and some gray candles and a whole heap of other
[25:20] candles so what you can do is you can use the setting so if if you're looking at your childart and thinking oh what Vector is
[25:30] that uh you know it doesn't look like a vector or whatever if you're using a different time frame what you can do is you can go to this box and say override with high time frame candles so if you
[25:40] check that it's going to change all of these candles to be what they are on the 4our chart so these are
[25:50] all um you know these are all um green vectors on on the FL hour so you chart ordinarily with this turned on but
[26:00] if you were using you know the higher time frame you just want to quickly verify you could either jump to the 4 Hour obviously or you could just use this box and that will save you having
[26:10] to go and actually you know draw the levels to verify you're in the right place so that's an option there um default settings for um
[26:20] overrides you don't need to change anything here um and we've got the same override with the binance equivalent
[26:30] like we have on the open interest and pvsr so don't change these and you'll never have a problem with having different candle colors to cabin crew
[26:40] charts again we have left the setting here to force the override if you want it however I can't think of a reason to recommend using it um at this stage um
[26:50] our EMAs if you want to hide some of them just uncheck the boxes and then finally we've got the market session
[27:00] info uh which displays up here it's the new Slimline version is the default if you want the old version uncheck the box and you're going to see the old bigger
[27:10] square that sort of spells things out a bit more and again you can change it to be displayed anywhere bottom
[27:20] right top left as you wish so the other thing to go through is the um alerts and we have uh quite a few
[27:30] custom ones on here so indicator three dots ADD alert and then when you expand
[27:40] this you'll see we've got alerts for the initial high of the day low of the day
[27:50] individually or we've got options for both so you could set one alert whether it's the high or the low being broken um
[28:00] you can get alert on that and we do the same for the daily levels the weekly levels 50% Asia 50% of the high of the
[28:10] day low of the day and the week uh EMA crosses just on the 200 the 50 and the 800 the
[28:20] pvsr uh Vector candle detected um so that will be any Vector candle and and then you've got a PVS crossing the 50
[28:30] EMA and then the US settle which is on the five and 15 minute time frame only so you have to go to the five or the 15
[28:40] minute chart to set that us hle alert um the other nice thing what we've done with the alerts is that they'll
[28:50] just carry on so you can set them now and you will always continue to get the alert for the ihod being crossed for example you don't have to reset those every week um if you want to have
[29:00] specific alerts um on just a green Vector for example what
[29:10] you can do on our PVS you set the alert on
[29:20] here so you can choose um only a bullish 200 so that means you'll only see green get getting
[29:30] alerts for for green vectors and again they can be set for as long as um your trading view
[29:40] subscription allows you to have alerts set for just a little bit of troubleshooting
[29:50] again to highlight there are a lot of calculations going on here so if you do have the indicator load and it show a red exclamation mark here you simply
[30:00] need to change to a different time frame and change back and that will allow to um recalculate the other thing that I want
[30:10] to show you is the support channel in Discord so what we have is the normal indicator support Channel um and then
[30:20] there's going to be a sticky message in there which says if you want it help with the Chrome extension or if you want to help with the indicator up to help
[30:30] desk and there's a whole bunch of frequently asked questions here um so we just ask that you do that first before
[30:40] using the the indicator support so you'll just come here to help disk and then you've got uh the Chrome section which
[30:50] obviously isn't about this video but there's a bunch of questions about the Chrome indicator and literally all you do is you um you click on one and the bot will think and then it will
[31:00] return the information to you so we've got the indicator sweet questions we got everything from how to find them and add them open interest um you know not
[31:10] working well it's not working because uh there's probably no data on the chart um as we went through um there's you know
[31:20] things on um on alerts um vectors you name it probably all the
[31:30] questions that we could preempt and then uh if nothing helps you go other and it will prompt you to go back to indicator support we ask if you can do as much
[31:40] description as possible and provide screenshots of the issue um that will get you to a solution um much faster
[31:50] that way otherwise I think that is about it um I hope you all enjoy
