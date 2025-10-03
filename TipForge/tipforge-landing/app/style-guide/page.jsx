"use client";

import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';

const StyleGuide = () => {
  const [copiedColor, setCopiedColor] = useState(null);

  const copyToClipboard = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopiedColor(id);
    setTimeout(() => setCopiedColor(null), 2000);
  };

  const colors = {
    primary: [
      { name: 'Dark Base', hex: '#1E1E1E', usage: 'Fő háttér' },
      { name: 'Darker', hex: '#121212', usage: 'Kártyák, szekciók' },
    ],
    secondary: [
      { name: 'Silver', hex: '#C0C0C0', usage: 'Szöveg, szeparátorok' },
      { name: 'Gray', hex: '#A9A9A9', usage: 'Placeholder, disabled' },
    ],
    accent: [
      { name: 'Neon Blue', hex: '#00D4FF', usage: 'CTA-k, kiemelések, linkek' },
      { name: 'Blue Dark', hex: '#00A8CC', usage: 'Hover állapotok' },
    ],
    feedback: [
      { name: 'Success Green', hex: '#00D98E', usage: 'Pozitív statisztikák, nyerő tippek' },
      { name: 'Warning Orange', hex: '#FF6B35', usage: 'Pain points, urgency, vesztő tippek' },
      { name: 'Info Blue', hex: '#3B82F6', usage: 'Informatív megjegyzések' },
    ]
  };

  const typography = [
    {
      name: 'H1 - Main Headline',
      font: 'Montserrat',
      weight: '700',
      size: '48px / 32px',
      lineHeight: '1.2',
      example: 'Nyerj, vagy ingyen folytatod'
    },
    {
      name: 'H2 - Section Title',
      font: 'Montserrat',
      weight: '600',
      size: '36px / 28px',
      lineHeight: '1.3',
      example: 'A TipForge Módszer'
    },
    {
      name: 'H3 - Subsection',
      font: 'Poppins',
      weight: '600',
      size: '24px / 20px',
      lineHeight: '1.4',
      example: 'Algoritmus, ami tényleg működik'
    },
    {
      name: 'Body Large',
      font: 'Inter',
      weight: '400',
      size: '18px / 16px',
      lineHeight: '1.6',
      example: 'Csatlakozz 600+ magyar fogadóhoz, akik otthagyták a drága bukó tippeket.'
    },
    {
      name: 'Body Regular',
      font: 'Inter',
      weight: '400',
      size: '16px / 16px',
      lineHeight: '1.6',
      example: 'Az első algoritmus-alapú tippszolgáltatás magyar fogadóknak.'
    },
    {
      name: 'Stats/Numbers',
      font: 'Roboto Mono',
      weight: '600',
      size: '32px / 24px',
      lineHeight: '1.2',
      example: '+124.300 Ft'
    }
  ];

  const buttons = [
    {
      name: 'Primary CTA',
      bg: '#00D4FF',
      text: '#1E1E1E',
      border: 'none',
      radius: '8px',
      padding: '16px 32px',
      example: 'Feliratkozom a várólistára'
    },
    {
      name: 'Secondary CTA',
      bg: 'transparent',
      text: '#00D4FF',
      border: '2px solid #00D4FF',
      radius: '8px',
      padding: '14px 30px',
      example: 'Csatlakozz a Discordra'
    },
    {
      name: 'Tertiary/Link',
      bg: 'transparent',
      text: '#00D4FF',
      border: 'none',
      radius: '0',
      padding: '0',
      example: '→ Tudj meg többet'
    }
  ];

  const ColorSwatch = ({ color, category }) => (
    <div className="flex items-center gap-3 p-3 bg-[#2A2A2A] rounded-lg hover:bg-[#333333] transition-colors">
      <div 
        className="w-16 h-16 rounded-lg shadow-lg flex-shrink-0"
        style={{ backgroundColor: color.hex }}
      />
      <div className="flex-1 min-w-0">
        <div className="font-semibold text-white">{color.name}</div>
        <div className="text-sm text-[#A9A9A9] truncate">{color.usage}</div>
      </div>
      <button
        onClick={() => copyToClipboard(color.hex, `${category}-${color.hex}`)}
        className="p-2 hover:bg-[#3A3A3A] rounded transition-colors flex-shrink-0"
      >
        {copiedColor === `${category}-${color.hex}` ? (
          <Check className="w-4 h-4 text-[#00D98E]" />
        ) : (
          <Copy className="w-4 h-4 text-[#C0C0C0]" />
        )}
      </button>
      <div className="text-sm font-mono text-[#00D4FF] flex-shrink-0">{color.hex}</div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#1E1E1E] text-white p-6">
      <div className="max-w-6xl mx-auto">
        
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-5xl font-bold mb-3 bg-gradient-to-r from-[#00D4FF] to-[#00A8CC] bg-clip-text text-transparent">
            TipForge Style Guide
          </h1>
          <p className="text-[#C0C0C0] text-lg">
            Vizuális identitás és design system a landing page-hez
          </p>
        </div>

        {/* Brand Attributes */}
        <section className="mb-12 p-6 bg-[#121212] rounded-xl border border-[#2A2A2A]">
          <h2 className="text-2xl font-bold mb-4 text-[#00D4FF]">Brand Attributes</h2>
          <div className="flex flex-wrap gap-3">
            {['Innovatív', 'Megbízható', 'Tech-savvy', 'Közösségi', 'Őszinte'].map(attr => (
              <span key={attr} className="px-4 py-2 bg-[#2A2A2A] rounded-full text-sm font-medium">
                {attr}
              </span>
            ))}
          </div>
        </section>

        {/* Color Palette */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6">Színpaletta</h2>
          
          <div className="space-y-6">
            <div>
              <h3 className="text-xl font-semibold mb-3 text-[#C0C0C0]">Primary (Háttér)</h3>
              <div className="space-y-2">
                {colors.primary.map(color => (
                  <ColorSwatch key={color.hex} color={color} category="primary" />
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-xl font-semibold mb-3 text-[#C0C0C0]">Secondary (Szöveg)</h3>
              <div className="space-y-2">
                {colors.secondary.map(color => (
                  <ColorSwatch key={color.hex} color={color} category="secondary" />
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-xl font-semibold mb-3 text-[#C0C0C0]">Accent (Kiemelés)</h3>
              <div className="space-y-2">
                {colors.accent.map(color => (
                  <ColorSwatch key={color.hex} color={color} category="accent" />
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-xl font-semibold mb-3 text-[#C0C0C0]">Feedback Colors</h3>
              <div className="space-y-2">
                {colors.feedback.map(color => (
                  <ColorSwatch key={color.hex} color={color} category="feedback" />
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Typography */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6">Tipográfia</h2>
          <div className="space-y-4">
            {typography.map(type => (
              <div key={type.name} className="p-6 bg-[#121212] rounded-xl border border-[#2A2A2A]">
                <div className="flex flex-wrap items-baseline justify-between mb-3 gap-4">
                  <h3 className="text-lg font-semibold text-[#00D4FF]">{type.name}</h3>
                  <div className="flex gap-6 text-sm text-[#A9A9A9]">
                    <span>Font: {type.font}</span>
                    <span>Weight: {type.weight}</span>
                    <span>Size: {type.size}</span>
                    <span>Line: {type.lineHeight}</span>
                  </div>
                </div>
                <div 
                  className="text-white"
                  style={{
                    fontFamily: type.font === 'Roboto Mono' ? 'monospace' : 'sans-serif',
                    fontWeight: type.weight,
                    fontSize: type.size.split('/')[0].trim(),
                    lineHeight: type.lineHeight
                  }}
                >
                  {type.example}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Buttons */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6">Button Styles</h2>
          <div className="space-y-6">
            {buttons.map(btn => (
              <div key={btn.name} className="p-6 bg-[#121212] rounded-xl border border-[#2A2A2A]">
                <h3 className="text-lg font-semibold text-[#00D4FF] mb-4">{btn.name}</h3>
                <div className="flex flex-wrap items-center gap-6">
                  <button
                    style={{
                      backgroundColor: btn.bg,
                      color: btn.text,
                      border: btn.border,
                      borderRadius: btn.radius,
                      padding: btn.padding,
                      fontWeight: '600',
                      fontSize: '16px',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease'
                    }}
                    className="hover:shadow-lg hover:shadow-[#00D4FF]/30"
                  >
                    {btn.example}
                  </button>
                  <div className="text-sm text-[#A9A9A9] space-y-1">
                    <div>BG: {btn.bg}</div>
                    <div>Text: {btn.text}</div>
                    <div>Border: {btn.border}</div>
                    <div>Radius: {btn.radius}</div>
                    <div>Padding: {btn.padding}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Spacing System */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6">Spacing System</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="p-6 bg-[#121212] rounded-xl border border-[#2A2A2A]">
              <h3 className="text-lg font-semibold text-[#00D4FF] mb-4">Section Spacing</h3>
              <div className="space-y-2 text-[#C0C0C0]">
                <div>Desktop padding: <span className="text-white font-mono">80px</span> top/bottom</div>
                <div>Mobile padding: <span className="text-white font-mono">48px</span> top/bottom</div>
                <div>Container max-width: <span className="text-white font-mono">1200px</span></div>
              </div>
            </div>
            <div className="p-6 bg-[#121212] rounded-xl border border-[#2A2A2A]">
              <h3 className="text-lg font-semibold text-[#00D4FF] mb-4">Element Spacing</h3>
              <div className="space-y-2 text-[#C0C0C0]">
                <div>Base grid: <span className="text-white font-mono">24px</span></div>
                <div>Mobile gutters: <span className="text-white font-mono">20px</span></div>
                <div>Min tap target: <span className="text-white font-mono">44px</span></div>
              </div>
            </div>
          </div>
        </section>

        {/* Icons & Images */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6">Ikonográfia & Képek</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="p-6 bg-[#121212] rounded-xl border border-[#2A2A2A]">
              <h3 className="text-lg font-semibold text-[#00D4FF] mb-4">Ikonok</h3>
              <ul className="space-y-2 text-[#C0C0C0]">
                <li>✓ Line icons (nem filled)</li>
                <li>✓ Neon kék stroke: <span className="text-[#00D4FF]">#00D4FF</span></li>
                <li>✓ 2px stroke vastagság</li>
                <li>✓ Min. 48×48px méret (mobile tap)</li>
                <li>✓ Lucide React library használható</li>
              </ul>
            </div>
            <div className="p-6 bg-[#121212] rounded-xl border border-[#2A2A2A]">
              <h3 className="text-lg font-semibold text-[#00D4FF] mb-4">Képek</h3>
              <ul className="space-y-2 text-[#C0C0C0]">
                <li>✓ Sötét overlay: 60% opacity</li>
                <li>✓ Mockup-ok neon kék accenttel</li>
                <li>✓ Minimalist chart/graph stílus</li>
                <li>✓ Kerüld a stock fotókat</li>
                <li>✓ Valódi testimonial-ok esetén high-quality</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Mobile Optimization */}
        <section className="mb-12 p-6 bg-[#121212] rounded-xl border border-[#2A2A2A]">
          <h2 className="text-2xl font-bold mb-4 text-[#00D4FF]">Mobile Optimization</h2>
          <div className="grid md:grid-cols-2 gap-4 text-[#C0C0C0]">
            <div>✓ Single column layout</div>
            <div>✓ Min. 44px tap targets</div>
            <div>✓ Sticky CTA bar (bottom)</div>
            <div>✓ Min. 16px font size</div>
            <div>✓ No hamburger menu (one-page)</div>
            <div>✓ Optimalizált képek</div>
          </div>
        </section>

        {/* Conversion Elements */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6">Konverziós Elemek</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {[
              { title: 'Progress Indicators', desc: 'Form lépésekhez, vizuális feedback' },
              { title: 'Micro-animations', desc: 'Scroll-ra, subtilis (nem zavaró)' },
              { title: 'Number Counters', desc: 'Animált számlálók statisztikákhoz' },
              { title: 'Social Proof Widgets', desc: 'Élő várolista számláló' }
            ].map(item => (
              <div key={item.title} className="p-6 bg-[#121212] rounded-xl border border-[#2A2A2A]">
                <h3 className="text-lg font-semibold text-white mb-2">{item.title}</h3>
                <p className="text-[#A9A9A9]">{item.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Code Example */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6">CSS Példa Kód</h2>
          <div className="p-6 bg-[#121212] rounded-xl border border-[#2A2A2A] overflow-x-auto">
            <pre className="text-sm text-[#C0C0C0] font-mono">
{`:root {
  /* Colors */
  --color-dark-base: #1E1E1E;
  --color-dark-darker: #121212;
  --color-silver: #C0C0C0;
  --color-gray: #A9A9A9;
  --color-neon-blue: #00D4FF;
  --color-blue-dark: #00A8CC;
  --color-success: #00D98E;
  --color-warning: #FF6B35;
  
  /* Spacing */
  --spacing-section-desktop: 80px;
  --spacing-section-mobile: 48px;
  --spacing-grid: 24px;
  --container-max: 1200px;
  
  /* Typography */
  --font-heading: 'Montserrat', sans-serif;
  --font-body: 'Inter', sans-serif;
  --font-mono: 'Roboto Mono', monospace;
}

.btn-primary {
  background: var(--color-neon-blue);
  color: var(--color-dark-base);
  border: none;
  border-radius: 8px;
  padding: 16px 32px;
  font-weight: 600;
  font-size: 18px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn-primary:hover {
  box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
  transform: translateY(-2px);
}`}
            </pre>
          </div>
        </section>

      </div>
    </div>
  );
};

export default StyleGuide;