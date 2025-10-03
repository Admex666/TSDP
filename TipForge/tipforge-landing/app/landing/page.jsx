'use client';

import React, { useState, useEffect } from 'react';
import { CheckCircle, TrendingUp, Users, Shield, BarChart3, MessageCircle, ArrowRight, X, ChevronDown } from 'lucide-react';

const TipForgeLanding = () => {
  const [email, setEmail] = useState('');
  const [showExitPopup, setShowExitPopup] = useState(false);
  const [waitlistCount, setWaitlistCount] = useState(287);
  const [activeTab, setActiveTab] = useState('algorithm');
  const [openFaq, setOpenFaq] = useState(null);

  // Exit intent detection
  useEffect(() => {
    const handleMouseLeave = (e) => {
      if (e.clientY <= 0 && !showExitPopup) {
        setShowExitPopup(true);
      }
    };
    document.addEventListener('mouseleave', handleMouseLeave);
    return () => document.removeEventListener('mouseleave', handleMouseLeave);
  }, [showExitPopup]);

  const handleSubmit = (e) => {
    e.preventDefault();
    // Google Analytics Event
    if (typeof window !== 'undefined' && window.gtag) {
      window.gtag('event', 'waitlist_signup', {
        'event_category': 'conversion',
        'event_label': 'hero_cta',
        'value': email
      });
    }
    alert(`Sikeres feliratkozás: ${email}`);
    setEmail('');
  };

  const testimonials = [
    { name: 'Kovács Dániel', age: 32, city: 'Budapest', text: '4 hónapig buktam havonta 20-30 ezret. Az első hónapban +47.000 Ft profitot csináltam.', avatar: '👨' },
    { name: 'Nagy Balázs', age: 28, city: 'Debrecen', text: 'Végre van akivel megbeszélni a tippeket, nem csak vakon rátenni.', avatar: '👨‍💼' },
    { name: 'Szabó Péter', age: 35, city: 'Szeged', text: 'Az algoritmus transzparens. Látod miért ajánlja. Ez benne a legjobb.', avatar: '🧑' }
  ];

  const stats = [
    { number: '127', label: 'Tipp kiadva', suffix: '' },
    { number: '73', label: 'Nyerő (57.5%)', suffix: '' },
    { number: '+12.4', label: 'ROI', suffix: '%' },
    { number: '+124', label: 'Profit (10k tétel)', suffix: 'k Ft' }
  ];

  const features = [
    {
      icon: <BarChart3 className="w-8 h-8" />,
      title: 'Algoritmus, ami működik',
      description: '15+ statisztikai tényező: xG, form, H2H, sérültek. Élő teljesítmény: 57.5% találati arány, +12.4% ROI',
      highlight: '127 tipp | 73 nyerő'
    },
    {
      icon: <Users className="w-8 h-8" />,
      title: 'Közösség, ami támogat',
      description: '600+ aktív fogadó Telegram + Discord-on. Élő meccs chat, közös elemzés, tapasztalt memberek mentorálása',
      highlight: 'Heti live Q&A'
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: '30 napos profitgarancia',
      description: 'Ha nem vagy profitban az első 30 nap végén, ingyen folytathatod amíg nem nyersz. Egyszerű, átlátható',
      highlight: 'Nulla kockázat'
    }
  ];

  const painPoints = [
    { icon: '📉', title: 'Havonta többet vesztesz, mint nyersz', text: 'Megint "biztos" volt a kupond. Az első 4 meccs is bejött. Az utolsó pedig... megint elvitte az egészet. Már megint -15.000 Ft a hónapban.' },
    { icon: '⚠️', title: 'Nem bízol a fizetős tippekben', text: 'Láttad már. Facebook-on hirdetik, "biztos nyerőket" ígérnek, aztán vagy nem jönnek be, vagy eltűnnek a pénzeddel. "Ez is csak egy újabb átverés lesz" - gondolod.' },
    { icon: '😤', title: 'Egyedül próbálod kitalálni', text: 'Órákig nézel statisztikákat, formát, sérültlistát. Úgy érzed, érted... de mégsem jön össze. Senki nem mondja meg, hol hibázol.' }
  ];

  const recentTips = [
    { date: 'Okt 2', match: 'Liverpool - Chelsea', tip: 'Over 2.5', odds: '1.85', result: '✅ 4-1', win: true },
    { date: 'Okt 2', match: 'Real - Atletico', tip: '1X', odds: '1.65', result: '❌ 0-1', win: false },
    { date: 'Okt 1', match: 'Bayern - Dortmund', tip: '1', odds: '1.72', result: '✅ 3-1', win: true },
    { date: 'Okt 1', match: 'PSG - Monaco', tip: 'BTTS', odds: '1.95', result: '✅ 3-2', win: true },
    { date: 'Szept 30', match: 'Arsenal - Tottenham', tip: '1', odds: '1.80', result: '✅ 2-0', win: true }
  ];

  const faqs = [
    { 
      q: '🤔 "Profitgarancia? Ez túl szép, hogy igaz legyen."',
      a: 'Értjük a kétséget. Ezért feltételekhez kötött: min. 20 tipp havonta, bankroll management betartása (1-2% tétek), min. 1.50 odds. Célja: biztosítsuk, hogy a rendszert követed, nem random fogadásokat teszel.'
    },
    { 
      q: '🔮 "Honnan tudom, hogy nem átverés?"',
      a: 'Teljes átláthatóság: élő eredménykövetés publikus spreadsheet-en, minden tipp látható előre és utólag. Discord közösség 600+ valódi emberrel. 30 napos garancia: ha nem nyersz, pénzt visszakapod.'
    },
    { 
      q: '🤖 "Miért algoritmus és nem emberi tipster?"',
      a: 'Emberek emocionálisak. Algoritmus objektív: 15+ statisztikai tényező, 5 év adat (50.000+ meccs). Nem tipp, hanem value bet azonosítás. De! Közösség elemzi, értelmezi - hibrid modell.'
    },
    { 
      q: '⏱️ "Mennyi időt kell rászánnom naponta?"',
      a: '5-10 perc. Reggel megkapod a napi 1-2 tippet Discord-on/email-ben. Elolvasod a rövid elemzést, ráteszed, kész. Opcionális: közösségi chat, de nem kötelező.'
    },
    { 
      q: '💰 "Mennyi tőke kell az induláshoz?"',
      a: 'Minimum 50.000 Ft ajánlott bankroll. 1-2% tétek = 500-1000 Ft/tipp. Kisebb is megy, de volatilitás miatt kockázatosabb. Nem tartozol elszámolással, saját számlád.'
    },
    { 
      q: '⚽ "Milyen sportokra adtok tippeket?"',
      a: 'Fő fókusz: futball (Premier League, La Liga, Bundesliga, Serie A, NB1). Bővülés terv: NBA, NHL, tenisz - de csak ha van elég adat a megbízható modellhez.'
    },
    { 
      q: '❌ "Mi van, ha nem működik?"',
      a: '30 napos profitgarancia. Ha veszteséges vagy, ingyen folytathatod következő hónapban. Ha továbbra sem nyersz, teljes visszatérítés. Nulla kockázat.'
    },
    { 
      q: '📅 "Mikor indul a szolgáltatás?"',
      a: 'Október 15. Launch. Várolista tagok 3 nappal korábban (okt 12) kapnak hozzáférést + launch árazás (2.990 Ft/hó vs. 4.990 Ft normál ár).'
    }
  ];

  return (
    <div className="min-h-screen bg-[#1E1E1E] text-white">
      
      {/* Hero Section */}
      <section className="relative overflow-hidden pt-20 pb-32 px-6">
        <div className="absolute inset-0 bg-gradient-to-b from-[#00D4FF]/10 to-transparent"></div>
        <div className="max-w-4xl mx-auto relative z-10">
          <div className="text-center mb-8">
            <div className="inline-block px-4 py-2 bg-[#FF6B35] rounded-full text-sm font-semibold mb-6 animate-pulse">
              🔥 300-ból már csak 47 hely maradt
            </div>
            <h1 className="text-5xl md:text-6xl font-bold mb-6 leading-tight">
              Nyerj, vagy <span className="text-[#00D4FF]">ingyen folytatod</span>
            </h1>
            <p className="text-xl text-[#C0C0C0] mb-4">
              30 napos profitgarancia magyar fogadóknak
            </p>
            <p className="text-lg text-[#A9A9A9] max-w-2xl mx-auto mb-12">
              Az első algoritmus-alapú tippszolgáltatás, ami <strong className="text-white">GARANTÁLJA</strong>, 
              hogy profitba forgatod a fogadásaidat - vagy addig ingyen megy, amíg nem nyersz.
            </p>
          </div>

          {/* Email Form */}
          <form onSubmit={handleSubmit} className="max-w-md mx-auto mb-8">
            <div className="flex gap-3">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="pelda@email.com"
                required
                className="flex-1 px-6 py-4 bg-[#2A2A2A] border-2 border-[#3A3A3A] rounded-lg focus:border-[#00D4FF] focus:outline-none text-white"
              />
              <button
                type="submit"
                className="px-8 py-4 bg-[#00D4FF] text-[#1E1E1E] font-bold rounded-lg hover:shadow-lg hover:shadow-[#00D4FF]/50 transition-all transform hover:scale-105"
              >
                Feliratkozom
              </button>
            </div>
            <p className="text-sm text-[#A9A9A9] mt-3 text-center">
              ✓ Nincs fizetési kötelezettség • ✓ Bármikor leiratkozhatsz
            </p>
          </form>

          {/* Social Proof */}
          <div className="flex items-center justify-center gap-3 text-sm text-[#C0C0C0]">
            <div className="flex -space-x-2">
              {['👨', '👩', '🧑', '👨‍💼', '👩‍💼'].map((emoji, i) => (
                <div key={i} className="w-10 h-10 rounded-full bg-[#2A2A2A] flex items-center justify-center border-2 border-[#1E1E1E] text-lg">
                  {emoji}
                </div>
              ))}
            </div>
            <p><strong className="text-white">{waitlistCount} ember</strong> már a várólistán</p>
          </div>

          {/* Benefits Pills */}
          <div className="flex flex-wrap justify-center gap-4 mt-12">
            <div className="flex items-center gap-2 px-4 py-2 bg-[#121212] rounded-full border border-[#2A2A2A]">
              <CheckCircle className="w-4 h-4 text-[#00D98E]" />
              <span className="text-sm">Algoritmus + közösség</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 bg-[#121212] rounded-full border border-[#2A2A2A]">
              <CheckCircle className="w-4 h-4 text-[#00D98E]" />
              <span className="text-sm">Élő eredménykövetés</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 bg-[#121212] rounded-full border border-[#2A2A2A]">
              <CheckCircle className="w-4 h-4 text-[#00D98E]" />
              <span className="text-sm">30 napos profitgarancia</span>
            </div>
          </div>
        </div>
      </section>

      {/* Problem Section */}
      <section className="py-20 px-6 bg-[#121212]">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold text-center mb-12">Ismerős?</h2>
          <div className="grid md:grid-cols-3 gap-6">
            {painPoints.map((pain, i) => (
              <div key={i} className="p-6 bg-[#1E1E1E] rounded-xl border border-[#2A2A2A] hover:border-[#FF6B35] transition-colors">
                <div className="text-4xl mb-4">{pain.icon}</div>
                <h3 className="text-xl font-semibold mb-3">{pain.title}</h3>
                <p className="text-[#A9A9A9] leading-relaxed">
                  {pain.text}
                </p>
              </div>
            ))}
          </div>

          <div className="mt-16 text-center max-w-2xl mx-auto">
            <p className="text-2xl mb-4">És a legrosszabb?</p>
            <p className="text-3xl font-bold text-[#FF6B35] mb-6">
              Azt hiszed, hogy <em>te vagy a hülye.</em>
            </p>
            <p className="text-lg text-[#C0C0C0] mb-4">
              Közben mindenki nyerésről posztol Facebookon. A haverjaid mesélnek arról, 
              hogy "megint nyertek 50 ezret". Te meg... te csak veszítesz.
            </p>
            <p className="text-xl mb-2">De a valóság?</p>
            <p className="text-2xl font-bold text-white mb-4">95% veszít hosszú távon.</p>
            <p className="text-lg text-[#A9A9A9]">
              Te nem vagy egyedül. A különbség a nyertesek és vesztesek között nem a szerencse.<br/>
              <strong className="text-[#00D4FF]">Hanem, hogy van-e rendszerük.</strong>
            </p>
          </div>
        </div>
      </section>

      {/* Solution Section */}
      <section className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">A TipForge Módszer</h2>
            <p className="text-xl text-[#C0C0C0]">Nem varázslat. Adat + Emberek + Védelem.</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, i) => (
              <div key={i} className="p-8 bg-[#121212] rounded-xl border-2 border-[#2A2A2A] hover:border-[#00D4FF] transition-all">
                <div className="text-[#00D4FF] mb-4">{feature.icon}</div>
                <h3 className="text-2xl font-bold mb-3">{feature.title}</h3>
                <p className="text-[#A9A9A9] mb-4 leading-relaxed">{feature.description}</p>
                <div className="px-4 py-2 bg-[#00D4FF]/10 rounded-lg inline-block">
                  <span className="text-[#00D4FF] font-semibold text-sm">{feature.highlight}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="py-20 px-6 bg-[#121212]">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold text-center mb-12">Mit mondanak a tagok?</h2>
          <div className="grid md:grid-cols-3 gap-6">
            {testimonials.map((test, i) => (
              <div key={i} className="p-6 bg-[#1E1E1E] rounded-xl border border-[#2A2A2A]">
                <div className="flex items-center gap-3 mb-4">
                  <div className="text-3xl">{test.avatar}</div>
                  <div>
                    <div className="font-semibold">{test.name}, {test.age}</div>
                    <div className="text-sm text-[#A9A9A9]">{test.city}</div>
                  </div>
                </div>
                <p className="text-[#C0C0C0] italic">"{test.text}"</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Transparency Section */}
      <section className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold text-center mb-4">
            Nem kérünk vak bizalmat. Mutassuk meg, hogyan működik.
          </h2>
          
          {/* Tabs */}
          <div className="flex justify-center gap-4 mb-8 flex-wrap">
            {[
              { id: 'algorithm', label: '🧮 Algoritmus' },
              { id: 'results', label: '📊 Eredmények' },
              { id: 'team', label: '👨‍💻 Csapat' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                  activeTab === tab.id 
                    ? 'bg-[#00D4FF] text-[#1E1E1E]' 
                    : 'bg-[#121212] text-[#C0C0C0] hover:bg-[#2A2A2A]'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="bg-[#121212] rounded-xl p-8 border border-[#2A2A2A]">
            {activeTab === 'algorithm' && (
              <div>
                <h3 className="text-2xl font-bold mb-6">Az Algoritmus Működése (emberi nyelven)</h3>
                <div className="space-y-6">
                  {[
                    { num: '1', title: 'Adat Begyűjtés', text: '15+ forrásból szedi össze az élő adatokat: expected goals (xG), shot accuracy, possession stats, form (utolsó 5 meccs), head-to-head mérleg, sérültek, motiváció index.' },
                    { num: '2', title: 'Mintázat Felismerés', text: '5 év történeti adat (~50.000 meccs) alapján tanult: "milyen csapat profilok nyernek adott szituációban". Pl.: high xG csapat rossz formában lévő ellen = value bet.' },
                    { num: '3', title: 'Érték Azonosítás', text: 'Összeveti a fogadóirodák oddsait az algoritmus által kalkulált "valós esélyekkel". Ha eltérés van = value bet. Példa: Valós esély 65%, de az odds 2.10-et ad (47.6% implied).' },
                    { num: '4', title: 'Confidence Score', text: 'Minden tipp kap egy 1-10 "magabiztosság" pontot. 8+ = high confidence (ajánlott követni). 5-7 = közepes (opcionális). Alatta = nem kerül kiadásra.' }
                  ].map(step => (
                    <div key={step.num} className="flex gap-4">
                      <div className="flex-shrink-0 w-12 h-12 bg-[#00D4FF] text-[#1E1E1E] rounded-full flex items-center justify-center font-bold text-xl">
                        {step.num}
                      </div>
                      <div>
                        <h4 className="text-xl font-semibold mb-2">{step.title}</h4>
                        <p className="text-[#A9A9A9]">{step.text}</p>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-8 p-4 bg-[#FF6B35]/10 border-l-4 border-[#FF6B35] rounded">
                  <p className="text-[#C0C0C0]">
                    ⚠️ <strong>Nem 100%-os.</strong> Hosszú távú előny a cél (55-60% win rate), 
                    nem minden tipp fog nyerni. A bankroll management kritikus.
                  </p>
                </div>
              </div>
            )}

            {activeTab === 'results' && (
              <div>
                <h3 className="text-2xl font-bold mb-6">Utolsó 30 Nap Eredményei (élő adatok)</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                  {stats.map((stat, i) => (
                    <div key={i} className="p-6 bg-[#1E1E1E] rounded-xl text-center border border-[#2A2A2A]">
                      <div className="text-4xl font-bold text-[#00D4FF] mb-2 font-mono">
                        {stat.number}{stat.suffix}
                      </div>
                      <div className="text-sm text-[#A9A9A9]">{stat.label}</div>
                    </div>
                  ))}
                </div>
                <p className="text-sm text-[#A9A9A9] mb-6">
                  🔴 Élő frissítés: Utolsó frissítés ma, 14:22
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-[#2A2A2A]">
                        <th className="text-left py-3 px-4 text-[#C0C0C0] font-semibold">Dátum</th>
                        <th className="text-left py-3 px-4 text-[#C0C0C0] font-semibold">Meccs</th>
                        <th className="text-left py-3 px-4 text-[#C0C0C0] font-semibold">Tipp</th>
                        <th className="text-left py-3 px-4 text-[#C0C0C0] font-semibold">Odds</th>
                        <th className="text-left py-3 px-4 text-[#C0C0C0] font-semibold">Eredmény</th>
                      </tr>
                    </thead>
                    <tbody>
                      {recentTips.map((tip, i) => (
                        <tr key={i} className="border-b border-[#2A2A2A] hover:bg-[#2A2A2A]/30">
                          <td className="py-3 px-4 text-[#A9A9A9]">{tip.date}</td>
                          <td className="py-3 px-4">{tip.match}</td>
                          <td className="py-3 px-4 font-semibold">{tip.tip}</td>
                          <td className="py-3 px-4 text-[#00D4FF] font-mono">{tip.odds}</td>
                          <td className={`py-3 px-4 font-semibold ${tip.win ? 'text-[#00D98E]' : 'text-[#FF6B35]'}`}>
                            {tip.result}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {activeTab === 'team' && (
              <div>
                <h3 className="text-2xl font-bold mb-6">Ki Csinálja?</h3>
                <div className="space-y-6">
                  <div className="flex gap-6 items-start">
                    <div className="w-20 h-20 bg-[#2A2A2A] rounded-full flex items-center justify-center text-4xl flex-shrink-0">
                      👨‍💻
                    </div>
                    <div>
                      <h4 className="text-xl font-semibold mb-2">Nagy Balázs – Founder & Adat Tudós</h4>
                      <p className="text-[#A9A9A9] mb-3">
                        8 éve foglalkozik sport analytics-szel. Korábban dolgozott data scientist pozícióban 
                        fintech startupnál. Mellette 10+ éve aktív sportfogadó – <strong className="text-white">ez egy personal pain megoldása is.</strong>
                      </p>
                      <p className="text-[#C0C0C0] italic">
                        "Elegem lett abból, hogy havonta bukjak. Gondoltam, tanulom adatelemzést munkahelyen, 
                        miért ne alkalmazhatnám fogadásra?"
                      </p>
                    </div>
                  </div>
                  <div className="flex gap-6 items-start">
                    <div className="w-20 h-20 bg-[#2A2A2A] rounded-full flex items-center justify-center text-4xl flex-shrink-0">
                      ⚽
                    </div>
                    <div>
                      <h4 className="text-xl font-semibold mb-2">Kovács Dávid – Community Manager</h4>
                      <p className="text-[#A9A9A9]">
                        Futball-megszállott, korábban futsal edző volt. Felel az élő közösségi támogatásért 
                        (Telegram/Discord), elemzésekért, és a napi tipp kommentárokért.
                      </p>
                    </div>
                  </div>
                </div>
                <div className="mt-8 p-4 bg-[#00D4FF]/10 border-l-4 border-[#00D4FF] rounded">
                  <p className="text-[#C0C0C0]">
                    💬 <strong>Elérhető vagyunk.</strong><br/>
                    Hétfő-Péntek 9-18 között élőben válaszolunk Discordon. 
                    Hétvégén meccsközvetítés alatt is aktívak vagyunk.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="py-20 px-6 bg-[#121212]">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-4xl font-bold text-center mb-4">Kérdések? Itt a válasz.</h2>
          <p className="text-center text-[#C0C0C0] mb-12">
            Tudjuk, hogy szkeptikus vagy. Jogosan. Mi is azok voltunk, amíg nem építettük meg ezt a rendszert.
          </p>
          <div className="space-y-4">
            {faqs.map((faq, i) => (
              <div key={i} className="bg-[#1E1E1E] rounded-xl border border-[#2A2A2A] overflow-hidden">
                <button
                  onClick={() => setOpenFaq(openFaq === i ? null : i)}
                  className="w-full px-6 py-4 flex justify-between items-center hover:bg-[#2A2A2A] transition-colors text-left"
                >
                  <span className="font-semibold text-lg">{faq.q}</span>
                  <ChevronDown className={`w-5 h-5 text-[#00D4FF] transition-transform ${openFaq === i ? 'rotate-180' : ''}`} />
                </button>
                {openFaq === i && (
                  <div className="px-6 pb-4 text-[#A9A9A9] leading-relaxed">
                    {faq.a}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold mb-6">
            Csatlakozz 600+ sportfogadóhoz
          </h2>
          <p className="text-xl text-[#C0C0C0] mb-8">
            Akik abbahagyták a találgatást, és elkezdtek tényleg nyerni
          </p>

          {/* Benefits */}
          <div className="grid md:grid-cols-3 gap-6 mb-12">
            <div className="p-6 bg-[#121212] rounded-xl border border-[#2A2A2A]">
              <div className="text-3xl mb-3">⚡</div>
              <h3 className="font-semibold mb-2">Korai hozzáférés</h3>
              <p className="text-sm text-[#A9A9A9]">3 nappal az indulás előtt (okt. 12.)</p>
            </div>
            <div className="p-6 bg-[#121212] rounded-xl border border-[#2A2A2A]">
              <div className="text-3xl mb-3">💰</div>
              <h3 className="font-semibold mb-2">Launch árazás</h3>
              <p className="text-sm text-[#A9A9A9]">3.990 Ft/hó (normál: 7.990 Ft)</p>
            </div>
            <div className="p-6 bg-[#121212] rounded-xl border border-[#2A2A2A]">
              <div className="text-3xl mb-3">🎁</div>
              <h3 className="font-semibold mb-2">Ajándék guide</h3>
              <p className="text-sm text-[#A9A9A9]">Bankroll management (érték: 9.990 Ft)</p>
            </div>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="max-w-md mx-auto mb-6">
            <div className="flex gap-3">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="pelda@email.com"
                required
                className="flex-1 px-6 py-4 bg-[#2A2A2A] border-2 border-[#3A3A3A] rounded-lg focus:border-[#00D4FF] focus:outline-none text-white"
              />
              <button
                type="submit"
                className="px-8 py-4 bg-[#00D4FF] text-[#1E1E1E] font-bold rounded-lg hover:shadow-lg hover:shadow-[#00D4FF]/50 transition-all transform hover:scale-105 flex items-center gap-2"
              >
                Feliratkozom
                <ArrowRight className="w-5 h-5" />
              </button>
            </div>
          </form>

          <p className="text-sm text-[#A9A9A9]">
            ✓ Nincs fizetési kötelezettség • ✓ Bármikor leiratkozhatsz • ✓ Email csak tippekre megy, spam 0%
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 bg-[#121212] border-t border-[#2A2A2A]">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <div>
              <div className="text-2xl font-bold text-[#00D4FF] mb-2">TipForge</div>
              <p className="text-sm text-[#A9A9A9]">Adat-alapú sportfogadás</p>
            </div>
            <div className="flex gap-8 text-sm">
              <a href="mailto:hello@tipforge.hu" className="text-[#C0C0C0] hover:text-[#00D4FF]">
                hello@tipforge.hu
              </a>
              <a href="#" className="text-[#C0C0C0] hover:text-[#00D4FF]">
                Discord
              </a>
              <a href="#" className="text-[#C0C0C0] hover:text-[#00D4FF]">
                Adatvédelem
              </a>
            </div>
          </div>
          <div className="mt-8 pt-8 border-t border-[#2A2A2A] text-center text-sm text-[#A9A9A9]">
            © 2025 TipForge. Minden jog fenntartva.
          </div>
        </div>
      </footer>

      {/* Exit Intent Popup */}
      {showExitPopup && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-6">
          <div className="bg-[#1E1E1E] rounded-xl max-w-md w-full p-8 relative border border-[#2A2A2A]">
            <button
              onClick={() => setShowExitPopup(false)}
              className="absolute top-4 right-4 text-[#A9A9A9] hover:text-white"
            >
              <X className="w-6 h-6" />
            </button>
            
            <h3 className="text-2xl font-bold mb-4">⚠️ Várj, mielőtt bezárod!</h3>
            <p className="text-[#C0C0C0] mb-4">
              A várolista <strong className="text-white">3 nap múlva bezár</strong>, és elveszíted:
            </p>
            <ul className="space-y-2 mb-6">
              <li className="flex items-start gap-2">
                <span className="text-[#FF6B35] mt-1">❌</span>
                <span className="text-[#A9A9A9]">-40% leárazást (3.990 Ft helyett 7.990 Ft-ra emelkedik)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-[#FF6B35] mt-1">❌</span>
                <span className="text-[#A9A9A9]">Korai hozzáférést (első tippek 3 nappal korábban)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-[#FF6B35] mt-1">❌</span>
                <span className="text-[#A9A9A9]">Ingyen bankroll guide-ot (9.990 Ft érték)</span>
              </li>
            </ul>
            <p className="text-[#C0C0C0] mb-4">
              Csak az <strong className="text-white">emailedet</strong> kérjük, nincs kötelezettség:
            </p>
            <form onSubmit={(e) => { handleSubmit(e); setShowExitPopup(false); }} className="mb-4">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="pelda@email.com"
                required
                className="w-full px-4 py-3 bg-[#2A2A2A] border border-[#3A3A3A] rounded-lg focus:border-[#00D4FF] focus:outline-none text-white mb-3"
              />
              <button
                type="submit"
                className="w-full px-6 py-3 bg-[#00D4FF] text-[#1E1E1E] font-bold rounded-lg hover:shadow-lg hover:shadow-[#00D4FF]/50 transition-all"
              >
                Igen, foglalom a helyem
              </button>
            </form>
            <button
              onClick={() => setShowExitPopup(false)}
              className="w-full text-sm text-[#A9A9A9] hover:text-white"
            >
              Nem, inkább fizetek teljes árat később
            </button>
          </div>
        </div>
      )}

      {/* Sticky Mobile CTA */}
      <div className="md:hidden fixed bottom-0 left-0 right-0 bg-[#1E1E1E] border-t border-[#2A2A2A] p-4 z-40 shadow-lg">
        <div className="flex items-center justify-between gap-3">
          <div className="flex-1">
            <div className="font-semibold text-sm">Csatlakozz {waitlistCount} emberhez</div>
            <div className="text-xs text-[#A9A9A9]">Várolista zárul okt 10-én</div>
          </div>
          <button
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            className="px-6 py-3 bg-[#00D4FF] text-[#1E1E1E] font-bold rounded-lg whitespace-nowrap text-sm"
          >
            Feliratkozom
          </button>
        </div>
      </div>
    </div>
  );
};

export default function LandingPage() {
  return <TipForgeLanding />;
}