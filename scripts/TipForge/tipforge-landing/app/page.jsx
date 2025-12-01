'use client';

import React, { useState, useEffect } from 'react';
import { CheckCircle, TrendingUp, Users, Shield, BarChart3, MessageCircle, ArrowRight, X, ChevronDown } from 'lucide-react';

const TipForgeLanding = () => {
  const [showExitPopup, setShowExitPopup] = useState(false);
  const [hasSeenExitPopup, setHasSeenExitPopup] = useState(false);
  const [waitlistCount, setWaitlistCount] = useState(9);
  const [activeTab, setActiveTab] = useState('algorithm');
  const [openFaq, setOpenFaq] = useState(null);

  // Tally widget betöltése
  useEffect(() => {
    const script = document.createElement('script');
    script.src = 'https://tally.so/widgets/embed.js';
    script.async = true;
    document.body.appendChild(script);
    
    return () => {
      if (document.body.contains(script)) {
        document.body.removeChild(script);
      }
    };
  }, []);

  // Exit intent detection
  useEffect(() => {
    const handleMouseLeave = (e) => {
      if (e.clientY <= 0 && !showExitPopup && !hasSeenExitPopup) {
        setShowExitPopup(true);
        setHasSeenExitPopup(true);
        
        if (typeof window !== 'undefined' && window.gtag) {
          window.gtag('event', 'exit_popup_shown', {
            'event_category': 'engagement',
          });
        }
      }
    };
    
    document.addEventListener('mouseleave', handleMouseLeave);
    return () => document.removeEventListener('mouseleave', handleMouseLeave);
  }, [showExitPopup, hasSeenExitPopup]);

  useEffect(() => {
    const handleScroll = () => {
      const scrollPercent = (window.scrollY / (document.documentElement.scrollHeight - window.innerHeight)) * 100;
      
      if (scrollPercent > 25 && !window.scrollTracked25) {
        window.scrollTracked25 = true;
        window.gtag?.('event', 'scroll_depth', {
          'event_category': 'engagement',
          'event_label': '25%'
        });
      }
      if (scrollPercent > 50 && !window.scrollTracked50) {
        window.scrollTracked50 = true;
        window.gtag?.('event', 'scroll_depth', {
          'event_category': 'engagement',
          'event_label': '50%'
        });
      }
      if (scrollPercent > 75 && !window.scrollTracked75) {
        window.scrollTracked75 = true;
        window.gtag?.('event', 'scroll_depth', {
          'event_category': 'engagement',
          'event_label': '75%'
        });
      }
      if (scrollPercent > 90 && !window.scrollTracked100) {
        window.scrollTracked100 = true;
        window.gtag?.('event', 'scroll_depth', {
          'event_category': 'engagement',
          'event_label': '100%'
        });
      }
    };
  
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleCloseExitPopup = () => {
    setShowExitPopup(false);
    
    if (typeof window !== 'undefined' && window.gtag) {
      window.gtag('event', 'exit_popup_dismissed', {
        'event_category': 'engagement',
      });
    }
  };

  const handleFaqClick = (index, question) => {
    const newOpenFaq = openFaq === index ? null : index;
    setOpenFaq(newOpenFaq);
    
    if (newOpenFaq !== null && typeof window !== 'undefined' && window.gtag) {
      window.gtag('event', 'faq_open', {
        'event_category': 'engagement',
        'event_label': question.substring(0, 50),
        'faq_index': index
      });
    }
  };

  const handleTabChange = (tabId) => {
    setActiveTab(tabId);
    
    if (typeof window !== 'undefined' && window.gtag) {
      window.gtag('event', 'tab_switch', {
        'event_category': 'engagement',
        'event_label': tabId
      });
    }
  };

  const handleWaitlistClick = (source) => {
    if (typeof window !== 'undefined' && window.gtag) {
      window.gtag('event', 'cta_click', {
        'event_category': 'engagement',
        'event_label': source,
        'source_section': source
      });
    }
    
    if (typeof window !== 'undefined' && window.Tally) {
      window.Tally.openPopup('wA1JkN', {
        layout: 'modal',
        width: 500,
        emoji: {
          text: '🔧',
          animation: 'wave'
        },
        onOpen: () => {
          if (window.gtag) {
            window.gtag('event', 'popup_open', {
              'event_category': 'conversion_funnel',
              'event_label': source
            });
          }
        },
        onSubmit: () => {
          if (window.gtag) {
            window.gtag('event', 'waitlist_signup', {
              'event_category': 'conversion',
              'event_label': source,
              'value': 1
            });
          }
        },
        onClose: () => {
          if (window.gtag) {
            window.gtag('event', 'popup_close', {
              'event_category': 'conversion_funnel',
              'event_label': source
            });
          }
        }
      });
    }
  };

  const testimonials = [
    { name: 'Kovács Dániel', age: 32, city: '', text: '4 hónapig buktam havonta 20-30 ezret. Végre van kivel megbeszélnem a tippeket.', avatar: '👨' },
    { name: 'Nagy Balázs', age: 28, city: '', text: 'Nem varázslat, de végre értem, hogy miért vesztettem annyit.', avatar: '👨‍💼' },
    { name: 'Szabó Péter', age: 35, city: '', text: 'A tippek jók, de az igazi érték a tudástár. Most már értem MIÉRT veszítettem korábban. Csak a tudás hiányzott – ilyen egyszerű.', avatar: '🧑' }
  ];

  const stats = [
    { number: '1326', label: 'Tipp', suffix: '' },
    { number: '862 (65%)', label: 'Nyertes', suffix: '' },
    { number: '+6', label: 'ROI', suffix: '%' },
    { number: '+130', label: 'Profit (10 ezer forintos tétekkel)', suffix: 'k Ft' }
  ];

  const features = [
    {
      icon: <BarChart3 className="w-8 h-8" />,
      title: 'Ne találgass. Legyél hosszútávon is nyertes.',
      description: '15+ statisztikai tényező: xG, forma, H2H, sérültek. Nem varázslat - józan ész és számok.',
      highlight: '127 tipp | 73 nyerő'
    },
    {
      icon: <Users className="w-8 h-8" />,
      title: 'Közösség, ahol tanulsz',
      description: 'Discord közösség tapasztalt fogadókkal. Élő meccs chat, közös elemzések, tanulás egymástól.',
      highlight: 'Stratégia megosztás'
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: 'Tanulj, ne csak kövess',
      description: '32 leckényi strukturált Tudástár. Minden modulban kvízek, gyakorlatok. Nem csak tippeket kapsz – megtanulod MIÉRT nyersz vagy vesztesz.',
      highlight: 'Kezdőből Profi 90 nap alatt'
    }
  ];

  const painPoints = [
    { icon: '📉', title: 'Találgatsz és többet vesztesz, mint nyersz', text: 'Megint "biztos" volt a szelód. Az első 4 meccs is bejött. Az utolsó pedig... megint elvitte az egészet. Már megint -15.000 Ft a hónapban.' },
    { icon: '⚠️', title: 'Nem tanulsz belőle', text: 'Követed a tippeket, de ha nyersz → nem tudod miért. Ha vesztesz → ugyanúgy nem tudod. Ez meggátol abban, hogy önálló legyél.' },
    { icon: '😤', title: 'Egyedül próbálod kitalálni', text: 'Órákig nézel statisztikákat, formát, sérültlistát. Úgy érzed, érted... de mégsem jön össze. Senki nem mondja meg, hol hibázol.' }
  ];

  const recentTips = [
    { date: '2025-11-30', match: 'Cleveland Cavaliers - Boston Celtics', tip: 'Vendég', odds: '3.45', result: '✅', win: true },
    { date: '2025-11-30', match: 'Utah Jazz - Houston Rockets', tip: 'Vendég', odds: '2.17', result: ' ✅', win: false },
    { date: '2025-11-29', match: 'Phoenix Suns - Denver Nuggets', tip: 'Hazai', odds: '2.37', result: '❌', win: true },
    { date: '2025-11-29', match: 'Miami Heat - Detroit Pistons', tip: 'Vendég', odds: '2.41', result: '✅', win: true },
  ];

  const faqs = [
    { 
      q: '🔮 "Honnan tudom, hogy nem átverés?"',
      a: 'Teljes átláthatóság: élő eredménykövetés publikus táblázaton, minden tipp látható előre és utólag. Discord közösség valódi emberekkel. Nem rejtünk semmit - ha buktunk, azt is látod. Ha nyertünk, azt is.'
    },
    { 
      q: '🤖 "Miért algoritmus és nem emberi tipster?"',
      a: 'Az emberek döntéseiben mindig vannak érzelmek. Az algoritmus objektív: 15+ statisztikai tényező, több év adat. De! Nem 100%-os - tévedés mindig lesz. A közösség segít értelmezni, mit miért ajánlunk. Hibrid modell: számok + emberi tapasztalat.'
    },
    { 
      q: '⏱️ "Mennyi időt kell rászánnom naponta?"',
      a: '5-10 perc. Reggel megkapod a napi 1-2 tippet Discord-on. Elolvasod a rövid elemzést, ráteszed, kész. Persze ott van a közösség, tananyagok, versenyek, ahol kis időráfordítással még jobb lehetsz.'
    },
    { 
      q: '💰 "Mennyi tőke kell az induláshoz?"',
      a: 'Minimum 50.000 Ft AJÁNLOTT bankroll. 1-2% tétek = 500-1000 Ft/tipp. Kisebb is megy, de volatilitás miatt kockázatosabb. Nem tartozol senkinek elszámolással, magad ura vagy és Te hozod meg a döntést.'
    },
    { 
      q: '⚽ "Milyen sportokra adtok tippeket?"',
      a: 'Jelenleg a fő fókusz: NBA és esport, de folyamatosan bővítjük újabb és újabb modellekkel és sportokkal a kínálatot.'
    },
    { 
      q: '📅 "Mikor indul a szolgáltatás?"',
      a: 'A tervezett indulás 2026. január 31. Várolista tagok 3 nappal korábban kapnak hozzáférést + kedvezményes árazás (előregisztrálóknak 4 990 Ft/hó vs. 7 990 Ft normál ár).'
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
              🔥 30-ből már csak 21 hely maradt
            </div>
            <h1 className="text-5xl md:text-6xl font-bold mb-6 leading-tight">
              Ne tippelgess. <span className="text-[#00D4FF]">Tanulj, építs rendszert, legyél nyertes.</span>
            </h1>
            <p className="text-xl text-[#C0C0C0] mb-4">
              Magyar sportfogadók közössége sporttudomány alapú tippekkel
            </p>
            <p className="text-lg text-[#A9A9A9] max-w-2xl mx-auto mb-12">
              Az első algoritmus-alapú tippszolgáltatás Magyarországon, ahol, a közösség aktív, és mindig tudsz tanulni, 
              hogyan legyél <strong className="text-white">még sikeresebb</strong>.
            </p>
          </div>

          {/* CTA Button - Hero */}
          <div className="max-w-md mx-auto mb-8">
          <button
            onClick={() => handleWaitlistClick('hero')}
            className="w-full px-8 py-5 bg-[#00D4FF] text-[#1E1E1E] text-lg font-bold rounded-lg hover:shadow-lg hover:shadow-[#00D4FF]/50 transition-all transform hover:scale-105"
          >
            Szeretnék hosszútávon nyertes lenni
          </button>
            <p className="text-sm text-[#A9A9A9] mt-3 text-center">
              ✓ Nincs fizetési kötelezettség • ✓ Bármikor leiratkozhatsz
            </p>
          </div>

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
              <span className="text-sm">Aktív közösség</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 bg-[#121212] rounded-full border border-[#2A2A2A]">
              <CheckCircle className="w-4 h-4 text-[#00D98E]" />
              <span className="text-sm">Egyedülálló tippek</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 bg-[#121212] rounded-full border border-[#2A2A2A]">
              <CheckCircle className="w-4 h-4 text-[#00D98E]" />
              <span className="text-sm">Egyedi tananyagok az azonnali fejlődéshez</span>
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
              Közben mindenki nagy nyereségekről posztol. A haverjaid mesélnek arról, 
              hogy "megint nyertek 50 ezret". Te meg... te csak veszítesz.
            </p>
            <p className="text-xl mb-2">De a valóság?</p>
            <p className="text-2xl font-bold text-white mb-4">95% veszít hosszú távon.</p>
            <p className="text-lg text-[#A9A9A9]">
              Nem vagy egyedül. A különbség a nyertesek és vesztesek között nem a szerencse.<br/>
              <strong className="text-[#00D4FF]">Hanem, hogy van-e rendszerük.</strong>
            </p>
          </div>
        </div>
      </section>

      {/* Solution Section */}
      <section className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">A <span className="text-[#00D4FF]">TipForge</span> Módszer</h2>
            <p className="text-xl text-[#C0C0C0]">Számok. Emberek. Transzparencia.</p>
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
            Teljes átláthatóság. Semmi rejtés.
          </h2>
          
          {/* Tabs */}
          <div className="flex justify-center gap-4 mb-8 flex-wrap">
            {[
              { id: 'algorithm', label: '🧮 Hogyan működik' },
              { id: 'results', label: '📊 Eredmények' },
              { id: 'team', label: '👨‍💻 Az Alapító' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => handleTabChange(tab.id)}
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
                <h3 className="text-2xl font-bold mb-6">Hogyan működik az algoritmus? (egyszerűen)</h3>
                <div className="space-y-6">
                  {[
                    { num: '1', title: 'Adatgyűjtés', text: 'Saját fejlesztésű rendszerünk 10+ forrásból szedi össze az emberi szem számára nem látható friss adatokat több sportról is.' },
                    { num: '2', title: 'Mintázatok felismerése', text: 'Több év adatait, több ezer meccset figyelembe véve tanult: olyan összefüggéseket ismer fel, amit veterán szakemberek sem.' },
                    { num: '3', title: 'Érték azonosítás', text: 'Összeveti a fogadóirodák oddsait az algoritmus által kalkulált esélyekkel, és a legjobb tippeket kiválasztja.' },
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
                    ⚠️ <strong>SEMMI SEM 100%-os.</strong> Jelenleg hosszú távú előnyünk van (akár 70% győzelmi arány), de 
                    nem minden tipp fog nyerni, semmi sem lehet tökéletes. A lényeg a hosszú távú győzelem.
                  </p>
                </div>
              </div>
            )}

            {activeTab === 'results' && (
              <div>
                <h3 className="text-2xl font-bold mb-6">Néhány eredményünk (NBA modell):</h3>
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
                      <h4 className="text-xl font-semibold mb-2">AJ – Alapító & Data Analyst</h4>
                      <p className="text-[#A9A9A9] mb-3">
                        Évek óta foglalkozik sporttal és sportelemzéssel. 5+ éve vagyok sportfogadó,
                        statisztikai-matematikai-programozási tudásomat pedig sikeresen használom a sportfogadás területén
                        – <strong className="text-white">a TipForge projekt egy személyes gondom megoldása is egyben.</strong>
                      </p>
                      <p className="text-[#C0C0C0] italic">
                        "Elegem lett abból, hogy havonta bukjak. Ha már tanultam adatelemzést egyetemen és munkahelyen is, 
                        miért ne alkalmaztam volna fogadásra is?"
                      </p>
                    </div>
                  </div>
                </div>
                <div className="mt-8 p-4 bg-[#00D4FF]/10 border-l-4 border-[#00D4FF] rounded">
                  <p className="text-[#C0C0C0]">
                    💬 <strong>Keress nyugodtan.</strong><br/>
                    Amint tudok, Discordon, illetve e-mailen is próbálok aktív lenni, keressetek nyugodtan: tipforgehq@gmail.com
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
                  onClick={() => handleFaqClick(i, faq.q)}
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
              <p className="text-sm text-[#A9A9A9]">3 nappal a hivatalos indulás előtt</p>
            </div>
            <div className="p-6 bg-[#121212] rounded-xl border border-[#2A2A2A]">
              <div className="text-3xl mb-3">💰</div>
              <h3 className="font-semibold mb-2">Launch árazás</h3>
              <p className="text-sm text-[#A9A9A9]">4 990 Ft/hó (napi mindössze 166 Ft)</p>
            </div>
            <div className="p-6 bg-[#121212] rounded-xl border border-[#2A2A2A]">
              <div className="text-3xl mb-3">🎁</div>
              <h3 className="font-semibold mb-2">Ajándék guide</h3>
              <p className="text-sm text-[#A9A9A9]">Bankroll management (érték: 9.990 Ft)</p>
            </div>
          </div>

          {/* Form */}
          <div className="max-w-md mx-auto mb-6">
          <button
            onClick={() => handleWaitlistClick('final_cta')}
            className="w-full px-8 py-5 bg-[#00D4FF] text-[#1E1E1E] text-lg font-bold rounded-lg hover:shadow-lg hover:shadow-[#00D4FF]/50 transition-all transform hover:scale-105 flex items-center justify-center gap-2"
          >
            Szeretnék hosszútávon nyertes lenni
            <ArrowRight className="w-5 h-5" />
          </button>
          </div>

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
              <p className="text-sm text-[#A9A9A9]">Adatalapú sportfogadó közösség</p>
            </div>
            <div className="flex gap-8 text-sm">
              <a href="mailto:tipforgehq@gmail.com" className="text-[#C0C0C0] hover:text-[#00D4FF]">
                tipforgehq@gmail.com
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
              onClick={handleCloseExitPopup}
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
                <span className="text-[#A9A9A9]">Indításkor 3000 Ft leárazást (4.990 Ft helyett 7.990 Ft-ra fog nőni)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-[#FF6B35] mt-1">❌</span>
                <span className="text-[#A9A9A9]">Korai hozzáférést (többi előfizetőhöz képest 3 nappal korábban kapsz tippeket)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-[#FF6B35] mt-1">❌</span>
                <span className="text-[#A9A9A9]">Ingyen bankroll guide-ot (9.990 Ft érték)</span>
              </li>
            </ul>
            <p className="text-[#C0C0C0] mb-4">
              Csak add meg az <strong className="text-white">emailedet</strong>, hogy ne maradj le a kedvezményről (NINCS fizetési kötelezettséged, bármikor visszaléphetsz):
            </p>
            <form onSubmit={(e) => { e.preventDefault(); handleWaitlistClick(); setShowExitPopup(false); }} className="mb-4">
            <button
              onClick={() => { handleWaitlistClick('exit_popup'); setShowExitPopup(false); }}
              className="w-full px-6 py-3 bg-[#00D4FF] text-[#1E1E1E] font-bold rounded-lg hover:shadow-lg hover:shadow-[#00D4FF]/50 transition-all"
            >
              Igen, szeretnék még több profitot!
            </button>
            </form>

            <button
            onClick={handleCloseExitPopup}
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
            onClick={() => handleWaitlistClick('mobile_sticky')}
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