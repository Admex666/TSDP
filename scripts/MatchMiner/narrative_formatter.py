import numpy as np
import pandas as pd
from config import PERCENTAGE_METRICS

# Hungarian translations of metrics for storytelling
METRIC_NAMES_HU = {
    # Team metrics
    "Goal": "szerzett gól",
    "xG": "várható gól (xG)",
    "xGOT": "kapura tartó lövések várható gólértéke (xGOT)",
    "xG Diff": "várható gólkülönbség (xG Diff)",
    "Poss%": "labdabirtoklás",
    "FldTilt": "field tilt (támadóharmadbeli labdabirtoklás)",
    "TchsA3": "támadóharmadbeli labdaérintés",
    "TchsA3%": "támadóharmadbeli labdaérintés arány",
    "TouchOpBox": "ellenfél tizenhatosán belüli labdaérintés",
    "BgChncCrtd": "kialakított nagy helyzet",
    "BgChnc": "nagy helyzet",
    "Pass%": "passzpontosság",
    "PsCmp": "sikeres passz",
    "PsAtt": "passzkísérlet",
    "AvgSeqPass": "szekvenciánkénti átlagos passzszám",
    "DirectSpeed": "támadások előrehaladási sebessége",
    "Sequences": "labdabirtoklási szekvencia",
    "Seq9+Pass": "legalább 9 passzból álló szekvencia",
    "PPDA": "PPDA (letámadás intenzitás)",
    "Tckl": "szerelés",
    "Int": "labdaszerzés / közbeavatkozás",
    "HighTurnovers": "magas labdaszerzés",
    "HighTurnoversOP": "magas labdaszerzés nyílt játékból",
    "HighTOEndShot": "magas labdaszerzésből született lövés",
    "HighTOEndGoal": "magas labdaszerzésből született gól",

    # Player metrics
    "GoalOP": "akciógól",
    "GoalSetPly": "pontrúgásból szerzett gól",
    "Shot": "lövés",
    "SOT": "kaput eltaláló lövés",
    "ShotConv": "lövésértékesítés",
    "OnTarget%": "kapura tartó lövések aránya",
    "BgChncMiss": "kihagyott nagy helyzet",
    "Ast": "gólpassz",
    "xA": "várható gólpassz (xA)",
    "KeyPass": "kulcspassz",
    "Passes in Final Third": "támadóharmadbeli passz",
    "PsIntoA3rd": "támadóharmadba juttatott passz",
    "ProgPass": "progresszív passz",
    "PsCmpInBox": "tizenhatoson belüli sikeres passz",
    "Crosses": "beadás",
    "CrossCmp": "sikeres beadás",
    "TakeOn": "cselkísérlet",
    "TakeOn%": "cselhatékonyság",
    "TakeOnFail": "sikertelen csel",
    "TakeOnSuccess": "sikeres csel",
    "TcklAtt": "szerelési kísérlet",
    "Recovery": "labdaszerzés (recovery)",
    "Clrnce": "felszabadítás",
    "ShtBlk": "blokkolt lövés",
    "BlkdPs": "blokkolt passz",
    "GrndDlWn": "megnyert földi párharc",
    "GrndDuels": "földi párharc",
    "AerialWon": "megnyert fejpárbaj",
    "Aerials": "fejpárbaj",
    "FoulCom": "elkövetett szabálytalanság",
    "Saves": "védés",
    "GoalCncd": "kapott gól",
    "GoalsPrev": "megelőzött gól (Goals Prevented)",
    "SvInBox": "tizenhatoson belüli védés",
    "CrossClaim": "lehúzott beadás",
    "CrossPunch": "kiöklözött beadás"
}

def format_value(val, metric_key):
    """Formats numeric values, appending % if the metric is percentage-based."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    
    # Check if percentage
    if "%" in metric_key or metric_key in ["OnTarget%", "ShotConv", "TakeOn%", "Pass%"]:
        return f"{val:.1f}%"
    
    # Integer formatting for count-based metrics
    if isinstance(val, (int, float)) and val == int(val):
        return f"{int(val)}"
        
    return f"{val:.2f}"

def format_team_insight(insight):
    """Generates a Hungarian HTML narrative for a team insight."""
    metric_hu = METRIC_NAMES_HU.get(insight['metric'], insight['metric'])
    val_str = format_value(insight['value'], insight['metric'])
    mean_str = format_value(insight['mean'], insight['metric'])
    
    direction = "átlag feletti" if insight['z_score'] > 0 else "átlag alatti"
    
    # Handle specific fields like PPDA where lower is actually more intense
    if insight['metric'] == 'PPDA':
        direction = "intenzívebb letámadás (alacsonyabb PPDA)" if insight['z_score'] < 0 else "kevésbé intenzív letámadás"
        
    narrative = (
        f"<strong>{insight['team']}</strong> a(z) <strong>{metric_hu}</strong> mutatóban "
        f"<strong>{val_str}</strong> értéket ért el a(z) <strong>{insight['opponent']}</strong> ellen "
        f"(torna átlag: {mean_str}). Ez egy kiemelkedően <strong>{direction}</strong> teljesítmény "
        f"(Z-score: {insight['z_score']:.2f})."
    )
    
    if insight['is_record']:
        narrative += " 🏆 <strong>Ez új torna-rekord!</strong>"
        
    return narrative

def format_player_insight(insight):
    """Generates a Hungarian HTML narrative for a player insight."""
    is_per90_active = insight.get('per90', False) and (insight['metric'] not in PERCENTAGE_METRICS)
    
    metric_hu = METRIC_NAMES_HU.get(insight['metric'], insight['metric'])
    if is_per90_active:
        metric_hu = f"{metric_hu} (90 percre vetítve)"
        
    val_str = format_value(insight['value'], insight['metric'])
    pos_mean_str = format_value(insight['pos_mean'], insight['metric'])
    
    direction = "átlag feletti" if insight['z_score_pos'] > 0 else "átlag alatti"
    
    narrative = (
        f"<strong>{insight['player']}</strong> ({insight['position']}, {insight['team']}) a(z) "
        f"<strong>{metric_hu}</strong> mutatóban <strong>{val_str}</strong> értéket produkált "
        f"a(z) <strong>{insight['opponent']}</strong> ellen. A posztján játszók torna átlaga: {pos_mean_str}. "
        f"Ez kiemelkedően <strong>{direction}</strong> ({insight['minutes']:.0f} játszott perc alatt, "
        f"Z-score poszthoz képest: {insight['z_score_pos']:.2f})."
    )
    
    # Add historical player comparison if available
    if 'z_score_player' in insight and not pd.isna(insight['z_score_player']):
        player_mean_str = format_value(insight['player_mean'], insight['metric'])
        p_dir = "saját átlaga felett" if insight['z_score_player'] > 0 else "saját átlaga alatt"
        narrative += f" A játékos saját korábbi átlaga ezen tornán {player_mean_str} volt (Z-score: {insight['z_score_player']:.2f}, {p_dir})."
        
    if insight['is_record']:
        narrative += " 🏆 <strong>Ez új torna-rekord!</strong>"
        
    return narrative
