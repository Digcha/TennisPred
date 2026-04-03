Data layout for this project:

- `tennis_atp-master/`
  ATP historical match files from Jeff Sackmann. This is the primary source used by `main.py`.
- `TennisDataMan/`
  Men's odds files from Tennis-Data. Reserved for future odds/market features.
- `TennisDataWomen/`
  Women's odds files. Not used by the current ATP prediction pipeline.

Current code paths:

- ATP matches: `DATA/tennis_atp-master`
- Men's odds: `DATA/TennisDataMan`
- Women's odds: `DATA/TennisDataWomen`

The prediction and benchmark commands currently train only on the ATP match history.
