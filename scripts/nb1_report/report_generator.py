"""
Automatizált Liga-Összefoglaló Generátor
Fő modul - report_generator.py
"""

import json
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from jinja2 import Environment, FileSystemLoader


class LigaReportGenerator:
    """
    Automatikus liga-összefoglaló PDF generátor
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializálás konfigurációs fájllal
        
        Args:
            config_path: A konfig YAML fájl elérési útja
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path(self.config['output']['directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Jinja2 környezet beállítása LaTeX-hez
        self.jinja_env = Environment(
            loader=FileSystemLoader('templates'),
            block_start_string='\\BLOCK{',
            block_end_string='}',
            variable_start_string='\\VAR{',
            variable_end_string='}',
            comment_start_string='\\#{',
            comment_end_string='}',
            line_statement_prefix='%%',
            line_comment_prefix='%#',
            trim_blocks=True,
            autoescape=False,
        )
    
    def load_data(self, data_path: str) -> Dict[str, Any]:
        """
        Betölti az input adatokat JSON fájlból
        
        Args:
            data_path: JSON fájl elérési útja
            
        Returns:
            Adatokat tartalmazó dictionary
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def calculate_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Kiszámítja az összes szükséges metrikát
        
        Args:
            data: Nyers input adatok
            
        Returns:
            Bővített adatok a számított metrikákkal
        """
        # DataFrame létrehozása a csapat adatokból
        teams_df = pd.DataFrame(data['teams'])
        
        # xPoints számítás (várható pontok xG alapján)
        teams_df['xPoints'] = self._calculate_xpoints(teams_df)
        
        # Forma index (súlyozott átlag az utolsó 5 meccsből)
        teams_df['form_index'] = teams_df['last_5_results'].apply(
            self._calculate_form_index
        )
        
        # Alul/felülteljesítés
        teams_df['performance_gap'] = teams_df['points'] - teams_df['xPoints']
        
        # Momentum (pozícióváltozás)
        teams_df['momentum'] = self._calculate_momentum(teams_df)
        
        # Trend (xG változás utolsó 6 fordulóban)
        teams_df['xg_trend'] = self._calculate_xg_trend(data)
        
        # Kategorizálás
        teams_df['category'] = teams_df.apply(self._categorize_team, axis=1)
        
        data['teams_metrics'] = teams_df.to_dict('records')
        
        # Mérkőzés metrikák
        data['matches_enhanced'] = self._enhance_matches(data['matches'])
        
        # Top listák
        data['top_stats'] = self._calculate_top_stats(teams_df)
        
        # Insights generálás
        data['insights'] = self._generate_insights(teams_df, data['matches'])
        
        return data
    
    def _calculate_xpoints(self, df: pd.DataFrame) -> pd.Series:
        """xPoints számítás xG és xGA alapján"""
        # Egyszerűsített modell: xG különbség alapú becslés
        xpoints = []
        for _, row in df.iterrows():
            xg_diff = row.get('xg_total', 0) - row.get('xga_total', 0)
            # Logisztikus függvény alkalmazása
            win_prob = 1 / (1 + pow(2.71828, -0.3 * xg_diff))
            draw_prob = 0.3 * (1 - abs(xg_diff) / 10)
            xp = win_prob * 3 + draw_prob * 1
            xpoints.append(round(xp * row.get('matches_played', 0), 1))
        return pd.Series(xpoints)
    
    def _calculate_form_index(self, results_string: str) -> float:
        """
        Forma index számítás az utolsó 5 meccsből
        Súlyozás: legutóbbi meccs a legsúlyosabb
        """
        if not results_string:
            return 0.0
        
        weights = [0.1, 0.15, 0.2, 0.25, 0.3]  # régebbi -> újabb
        points_map = {'W': 3, 'D': 1, 'L': 0}
        
        results = list(results_string)[-5:]  # utolsó 5
        weighted_sum = sum(
            points_map.get(r, 0) * weights[i] 
            for i, r in enumerate(results)
        )
        
        return round(weighted_sum / sum(weights[:len(results)]), 2)
    
    def _calculate_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Pozícióváltozás az utolsó fordulókban"""
        momentum = []
        for _, row in df.iterrows():
            positions = row.get('position_history', [])
            if len(positions) >= 2:
                change = positions[-2] - positions[-1]  # pozitív = felfelé
                momentum.append(change)
            else:
                momentum.append(0)
        return pd.Series(momentum)
    
    def _calculate_xg_trend(self, data: Dict) -> List[float]:
        """xG trend számítás (dummy implementáció)"""
        return [0.0] * len(data['teams'])
    
    def _categorize_team(self, row: pd.Series) -> str:
        """Csapat kategorizálása teljesítmény alapján"""
        form = row['form_index']
        position = row['position']
        gap = row['performance_gap']
        
        thresholds = self.config['thresholds']
        
        if form >= thresholds['good_form'] and gap > 0:
            return 'upward'
        elif form < thresholds['bad_form'] and position >= thresholds['relegation_zone']:
            return 'danger'
        elif gap > thresholds['overperforming']:
            return 'overperforming'
        elif gap < -thresholds['underperforming']:
            return 'underperforming'
        else:
            return 'stable'
    
    def _enhance_matches(self, matches: List[Dict]) -> List[Dict]:
        """Mérkőzés adatok bővítése számított mezőkkel"""
        enhanced = []
        for match in matches:
            m = match.copy()
            
            # Performance gap
            home_xg = m.get('home_xg', 0)
            away_xg = m.get('away_xg', 0)
            home_goals = m.get('home_goals', 0)
            away_goals = m.get('away_goals', 0)
            
            m['xg_diff'] = home_xg - away_xg
            m['goal_diff'] = home_goals - away_goals
            m['surprise_factor'] = abs(m['xg_diff'] - m['goal_diff'])
            
            # Kategorizálás
            if m['surprise_factor'] > 2:
                m['match_type'] = 'surprise'
            else:
                m['match_type'] = 'expected'
            
            enhanced.append(m)
        
        return enhanced
    
    def _calculate_top_stats(self, df: pd.DataFrame) -> Dict[str, List]:
        """Top 5 listák generálása különböző kategóriákban"""
        return {
            'top_attack': df.nlargest(5, 'xg_total')[['team_name', 'xg_total']].to_dict('records'),
            'top_defense': df.nsmallest(5, 'xga_total')[['team_name', 'xga_total']].to_dict('records'),
            'best_form': df.nlargest(5, 'form_index')[['team_name', 'form_index']].to_dict('records'),
            'worst_form': df.nsmallest(5, 'form_index')[['team_name', 'form_index']].to_dict('records'),
            'overperformers': df.nlargest(5, 'performance_gap')[['team_name', 'performance_gap']].to_dict('records'),
            'underperformers': df.nsmallest(5, 'performance_gap')[['team_name', 'performance_gap']].to_dict('records'),
        }
    
    def _generate_insights(self, df: pd.DataFrame, matches: List[Dict]) -> List[str]:
        """Automatikus insight generálás"""
        insights = []
        
        # Legjobb forma
        best_form_team = df.nlargest(1, 'form_index').iloc[0]
        insights.append(
            f"{best_form_team['team_name']} kiváló formában: "
            f"{best_form_team['form_index']:.1f} forma-index, "
            f"{best_form_team['position']}. helyezés"
        )
        
        # Veszélyzóna
        danger_teams = df[df['category'] == 'danger']
        if not danger_teams.empty:
            team = danger_teams.iloc[0]
            insights.append(
                f"{team['team_name']} veszélyes zónában: "
                f"kiesőhelyen, gyenge forma ({team['form_index']:.1f})"
            )
        
        # Meglepetés eredmény
        surprise_matches = [m for m in matches if m.get('surprise_factor', 0) > 2]
        if surprise_matches:
            m = surprise_matches[0]
            insights.append(
                f"Meglepetés: {m['home_team']} vs {m['away_team']} - "
                f"xG alapú várakozástól eltérő eredmény"
            )
        
        return insights[:5]  # Maximum 5 insight
    
    def generate_latex(self, data: Dict[str, Any]) -> str:
        """
        LaTeX dokumentum generálása Jinja2 template alapján
        
        Args:
            data: Feldolgozott adatok
            
        Returns:
            LaTeX forráskód string
        """
        template = self.jinja_env.get_template('main_template.tex')
        
        # Színkódok hozzáadása
        data['colors'] = self.config['colors']
        data['zone_limits'] = self.config['zone_limits']
        
        # Generálás dátuma
        data['generated_date'] = datetime.now().strftime('%Y. %B %d.')
        
        return template.render(**data)
    
    def compile_pdf(self, latex_content: str, output_name: str) -> Path:
        """
        LaTeX fordítása PDF-é
        
        Args:
            latex_content: LaTeX forráskód
            output_name: Kimeneti fájl neve (kiterjesztés nélkül)
            
        Returns:
            Generált PDF fájl elérési útja
        """
        # LaTeX fájl írása
        tex_path = self.output_dir / f"{output_name}.tex"
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        # PDF fordítás
        try:
            subprocess.run(
                ['pdflatex', '-output-directory', str(self.output_dir), str(tex_path)],
                check=True,
                capture_output=True
            )
            # Második futtatás a hivatkozásokhoz
            subprocess.run(
                ['pdflatex', '-output-directory', str(self.output_dir), str(tex_path)],
                check=True,
                capture_output=True
            )
            
            pdf_path = self.output_dir / f"{output_name}.pdf"
            print(f"✓ PDF sikeresen generálva: {pdf_path}")
            return pdf_path
            
        except subprocess.CalledProcessError as e:
            print(f"✗ LaTeX fordítási hiba: {e}")
            raise
    
    def generate_report(self, data_path: str, output_name: str = None) -> Path:
        """
        Teljes riport generálási folyamat
        
        Args:
            data_path: Input adatok JSON fájlja
            output_name: Kimeneti fájl neve (opcionális)
            
        Returns:
            Generált PDF elérési útja
        """
        print("=" * 60)
        print("Liga-Összefoglaló Generátor")
        print("=" * 60)
        
        # 1. Adatok betöltése
        print("\n[1/4] Adatok betöltése...")
        data = self.load_data(data_path)
        
        # 2. Metrikák számítása
        print("[2/4] Metrikák számítása...")
        data = self.calculate_metrics(data)
        
        # 3. LaTeX generálás
        print("[3/4] LaTeX dokumentum generálása...")
        latex_content = self.generate_latex(data)
        
        # 4. PDF fordítás
        print("[4/4] PDF fordítás...")
        if output_name is None:
            output_name = f"liga_report_round_{data['round_number']}"
        
        pdf_path = self.compile_pdf(latex_content, output_name)
        
        print("\n" + "=" * 60)
        print("✓ Riport sikeresen elkészült!")
        print("=" * 60)
        
        return pdf_path


# Példa használat
if __name__ == "__main__":
    generator = LigaReportGenerator("config.yaml")
    generator.generate_report("data/round_10.json")