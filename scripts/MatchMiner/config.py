# Mapping configurations for Team and Player metrics.
# Key is the user-friendly name / ID, value is the exact column name in the CSV.

TEAM_METRICS = {
    # Offense & Possession
    "Goal": "Goal",
    "xG": "xG",
    "xGOT": "xGOT",
    "xG Diff": "xG Diff",
    "Poss%": "Poss%",
    "FldTilt": "FldTilt",
    "TchsA3": "TchsA3",
    "TchsA3%": "TchsA3%",
    "TouchOpBox": "TouchOpBox",
    "BgChncCrtd": "BgChncCrtd",
    "BgChnc": "BgChnc",
    
    # Passing & Style
    "Pass%": "Pass%",
    "PsCmp": "PsCmp",
    "PsAtt": "PsAtt",
    "AvgSeqPass": "AvgSeqPass",
    "DirectSpeed": "DirectSpeed",
    "Sequences": "Sequences",
    "Seq9+Pass": "Seq9+Pass",
    
    # Defending & Pressing
    "PPDA": "PPDA",
    "Tckl": "Tckl",
    "Int": "Int",
    "HighTurnovers": "HighTurnovers",
    "HighTurnoversOP": "HighTurnoversOP",
    "HighTOEndShot": "HighTOEndShot",
    "HighTOEndGoal": "HighTOEndGoal",
}

PLAYER_METRICS = {
    # Offense & Shooting
    "Goal": "Goal",
    "GoalOP": "GoalOP",
    "GoalSetPly": "GoalSetPly",
    "xG": "xG",
    "xGOT": "xGOT",
    "Shot": "Shot",
    "SOT": "SOT",
    "ShotConv": "ShotConv",
    "OnTarget%": "OnTarget%",
    "TouchOpBox": "TouchOpBox",
    "BgChnc": "BgChnc",
    "BgChncMiss": "BgChncMiss",
    
    # Passing & Playmaking
    "Ast": "Ast",
    "xA": "xA",
    "KeyPass": "KeyPass",
    "BgChncCrtd": "BgChncCrtd",
    "Passes in Final Third": "Passes in the Final Third",
    "PsIntoA3rd": "PsIntoA3rd",
    "ProgPass": "ProgPass",
    "PsCmpInBox": "PsCmpInBox",
    "PsCmp": "PsCmp",
    "PsAtt": "PsAtt",
    "Pass%": "Pass%",
    "Crosses": "Crosses",
    "CrossCmp": "CrossCmp",
    
    # Take-ons (Dribbles)
    "TakeOn": "TakeOn",
    "TakeOn%": "TakeOn%",
    "TakeOnFail": "TakeOnFail",
    "TakeOnSuccess": "TakeOnSuccess",  # Derived field: TakeOn - TakeOnFail
    
    # Defending
    "Tckl": "Tckl",
    "TcklAtt": "TcklAtt",
    "Int": "Int",
    "Recovery": "Recovery",
    "Clrnce": "Clrnce",
    "ShtBlk": "ShtBlk",
    "BlkdPs": "BlkdPs",
    "GrndDlWn": "GrndDlWn",
    "GrndDuels": "GrndDuels",
    "AerialWon": "AerialWon",
    "Aerials": "Aerials",
    "FoulCom": "FoulCom",
    
    # Goalkeeping
    "Saves": "Saves",
    "GoalCncd": "GoalCncd",
    "GoalsPrev": "GoalsPrev",
    "SvInBox": "SvInBox",
    "CrossClaim": "CrossClaim",
    "CrossPunch": "CrossPunch"
}

# Metrics that are percentage-based or ratios, so we shouldn't use simple volume comparisons
PERCENTAGE_METRICS = {
    "Poss%", "FldTilt", "TchsA3%", "Pass%", "PPDA", "DirectSpeed", "AvgSeqPass",
    "OnTarget%", "ShotConv", "TakeOn%", "MinPerGoal", "%ShotHead", "%ShotInBox", "%ShtOutBox"
}

# Default scoring weights
WEIGHTS = {
    "rarity": 0.5,      # Weight for absolute Z-score (deviation from baseline)
    "record": 0.3,      # Weight if the performance is a tournament/historical record
    "context": 0.2      # Weight based on game context (e.g. substitute player, knockout stage)
}
