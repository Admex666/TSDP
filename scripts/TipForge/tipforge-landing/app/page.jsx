'use client';

import React, { useState, useEffect } from 'react';
import { CheckCircle, TrendingUp, Users, Shield, BarChart3, MessageCircle, ArrowRight, X, ChevronDown } from 'lucide-react';

const TipForgeLanding = () => {
  const [showExitPopup, setShowExitPopup] = useState(false);
  const [hasSeenExitPopup, setHasSeenExitPopup] = useState(false);
  const [waitlistCount, setWaitlistCount] = useState(23);
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

        if (typeof window !== 'undefined' && window.fbq) {
          window.fbq('track', 'ViewContent', { content_name: 'exit_popup' });
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

    if (typeof window !== 'undefined' && window.fbq) {
      window.fbq('track', 'Contact', { content_name: source });
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
          if (window.fbq) {
            window.fbq('track', 'InitiateCheckout', { content_name: source });
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
          if (window.fbq) {
            window.fbq('track', 'Lead', { content_name: source });
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
    { name: 'Tamás', age: 'Kezdő', city: '', text: '3 hónap alatt +18% ROI-t értem el az első hónaptól az előre definiált bankroll-tervvel.', avatar: '👨' },
    { name: 'Anna', age: 'Kezdő', city: '', text: 'Az első hét után +12% profitot könyveltem el, miközben minimális erőfeszítéssel követtem az automatizált tippeket.', avatar: '👩' },
    { name: 'Gábor', age: 'Haladó', city: '', text: 'A Telegram csoport segítségével optimalizáltam a stratégiámat és 3 hónap alatt +25% ROI-t értem el.', avatar: '🧑' }
  ];

  const stats = [
    { number: '15+', label: 'Statisztikai mutató', suffix: '' },
    { number: '70', label: 'Tipp sikeresség (Béta)', suffix: '%' },
    { number: '11', label: 'Átlagos ROI (3 hó)', suffix: '%' },
    { number: '10', label: 'Perc napi ráfordítás', suffix: '' }
  ];

  const features = [
    {
      icon: <BarChart3 className="w-8 h-8" />,
      title: 'Adatvezérelt döntések',
      description: 'Elemzések, statisztikák és valós adatok alapján mutatjuk a legjobb tippeket. Nincs több vakon fogadás.',
      highlight: 'Elemzett statisztika'
    },
    {
      icon: <Users className="w-8 h-8" />,
      title: 'Automatizált tippkövetés',
      description: 'Az előre definiált bankroll-terv és az automatizált rendszer segítségével könnyedén követheted a stratégiát.',
      highlight: 'Minimális erőfeszítés'
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: 'Garantált fejlődés',
      description: 'Edukációs tartalmakkal segítünk megérteni a miérteket. Ha nincs javulás a nyereségedben, visszafizetjük a díjad.',
      highlight: 'Kockázatmentes'
    }
  ];

  const painPoints = [
    { icon: '📉', title: 'Vak fogadások vesztesége', text: 'Tudtad, hogy a legtöbb fogadó még mindig véletlenszerű tippekre hagyatkozik? Ez hónapról hónapra csak pénzt veszít neked.' },
    { icon: '⚠️', title: 'Hiányzó ROI követés', text: 'Hány fogadó tartja számon a saját ROI-ját? A válasz: kevesen. Statisztika nélkül csak sötétben tapogatózol.' },
    { icon: '😤', title: 'Szerencsejáték vs. Tudás', text: 'Miért bukik el 90% a sportfogadókból? Mert a szerencsére építenek, nem a megalapozott statisztikai háttérre.' }
  ];

  const recentTips = [
    { date: '2026-01-22', match: 'LA Lakers - GS Warriors', tip: 'Hazai', odds: '1.85', result: '✅', win: true },
    { date: '2026-01-22', match: 'Milwaukee Bucks - Boston Celtics', tip: 'Vendég', odds: '2.10', result: '✅', win: true },
    { date: '2026-01-22', match: 'Dallas Mavericks - Phoenix Suns', tip: 'Hazai', odds: '1.75', result: '✅', win: true },
    { date: '2026-01-22', match: 'Denver Nuggets - LA Clippers', tip: 'Vendég', odds: '2.45', result: '❌', win: false },
  ];

  const faqs = [
    {
      q: '⚽ "Miért adatalapú a TipForge?"',
      a: 'Mert a profi fogadók 80%-a nem a szerencsére, hanem statisztikára épít. Mi minden tipphez mutatjuk a statisztikai hátteret, hogy tudd, miért megalapozott a döntés.'
    },
    {
      q: '💰 "Mennyi időt kell rászánnom?"',
      a: 'Napi 10-15 percet. Az automatizált tippkövetés és az előre definiált bankroll-terv segítségével minimális erőfeszítéssel követheted a stratégiát.'
    },
    {
      q: '🛡️ "Mi a garancia a nyereségre?"',
      a: 'Ha az első hónapod végén 5%-nál nagyobb a tippeken realizált veszteség, automatikusan jóváírunk neked egy extra hónapot – így azokat a tippeket ingyen, kockázat nélkül kapod.'
    },
    {
      q: '📈 "Mit jelent a ROI és miért fontos?"',
      a: 'A Return on Investment megmutatja a befektetett tőkéd arányos nyereségét. Ügyfeleink az elmúlt 3 hónapban átlagosan 11%-os ROI-t értek el.'
    },
    {
      q: '🤝 "Milyen közösséghez csatlakozom?"',
      a: 'Egy olyan csapathoz, ahol a profit és a tanulás kéz a kézben jár. Edukációs bejegyzéseinkből megtanulod a fogadás logikáját is, nem csak a tippeket kapod.'
    },
    {
      q: '📅 "Mikor indul a kedvezményes ár?"',
      a: 'A béta program keretében most limitált létszám mellé 7 990 Ft helyett mindössze 5 490 Ft-ért csatlakozhatsz.'
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
              🔥 30-ból már csak {30 - waitlistCount} hely maradt
            </div>
            <h1 className="text-5xl md:text-6xl font-bold mb-6 leading-tight">
              Fogadj <span className="text-[#00D4FF]">megalapozottan, ne vakon.</span>
            </h1>
            <p className="text-xl text-[#C0C0C0] mb-4">
              Hiteles, adatvezérelt sports betting tanácsadás a hosszútávú nyereségért.
            </p>
            <p className="text-lg text-[#A9A9A9] max-w-2xl mx-auto mb-12">
              Tudtad, hogy a profi fogadók 80%-a <strong className="text-white">nem a szerencsére</strong>, hanem statisztikára épít?
              Mutatjuk, hogyan érhetsz el stabil nyereséget adatalapú tippekkel.
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
              <span className="text-sm">Garancia</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 bg-[#121212] rounded-full border border-[#2A2A2A]">
              <CheckCircle className="w-4 h-4 text-[#00D98E]" />
              <span className="text-sm">Egyedülálló tippek</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 bg-[#121212] rounded-full border border-[#2A2A2A]">
              <CheckCircle className="w-4 h-4 text-[#00D98E]" />
              <span className="text-sm">Személyre szabott fejlesztés</span>
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
              Közben mindenki nagy nyereségekről posztol. Azt hiszed, hogy csak a szerencsén múlik.
              Te meg... te csak veszítesz minden hónapban vak fogadásokon.
            </p>
            <p className="text-xl mb-2">De a valóság?</p>
            <p className="text-2xl font-bold text-white mb-4">95% veszít hosszú távon.</p>
            <p className="text-lg text-[#A9A9A9]">
              Nem vagy egyedül. A különbség a nyertesek és vesztesek között nem a szerencse.<br />
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

          <div className="grid md:grid-cols-3 gap-8 mb-20">
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

          {/* Detailed Offer Section */}
          <div className="max-w-4xl mx-auto p-10 bg-[#121212] rounded-2xl border border-[#00D4FF]/30 relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4">
              <div className="bg-[#00D4FF] text-[#1E1E1E] px-4 py-1 rounded-full text-sm font-bold uppercase tracking-wider">
                Limitált Ajánlat
              </div>
            </div>
            <h3 className="text-3xl font-bold mb-6 text-white">Csatlakozz hozzánk, és fogadj okosan, ne vakon!</h3>
            <div className="space-y-4 text-lg text-[#C0C0C0] leading-relaxed">
              <p>
                Ügyfeleink az elmúlt 3 hónapban átlagosan <strong className="text-white">11%-os ROI-t</strong> értek el,
                miközben az első napoktól azonnal kipróbálható tippeket kapsz.
              </p>
              <p>
                Az előre definiált bankroll-terv és az automatizált tippkövetés segítségével könnyedén követheted
                a stratégiát, minimális erőfeszítéssel.
              </p>
              <div className="p-6 bg-[#00D4FF]/5 border border-[#00D4FF]/20 rounded-xl my-6">
                <p className="text-white font-semibold flex items-center gap-3">
                  <Shield className="w-6 h-6 text-[#00D4FF]" />
                  ROI GARANCIA:
                </p>
                <p className="mt-2 italic">
                  "Ha az első hónapod végén 5%-nál nagyobb a tippeken realizált veszteség,
                  automatikusan jóváírunk neked egy extra hónapot – így azokat a tippeket ingyen, kockázat nélkül kapod."
                </p>
              </div>
              <p className="text-xl text-white font-bold">
                Tudd meg, hogyan válhat a fogadás szerencsejátékból stabil, nyereséges tudássá!
              </p>
            </div>
            <div className="mt-10">
              <button
                onClick={() => handleWaitlistClick('offer_section')}
                className="w-full md:w-auto px-10 py-5 bg-[#00D4FF] text-[#1E1E1E] text-xl font-bold rounded-xl hover:shadow-2xl hover:shadow-[#00D4FF]/40 transition-all flex items-center justify-center gap-3"
              >
                Kérem a béta hozzáférést
                <ArrowRight className="w-6 h-6" />
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Lead Magnet Section */}
      <section className="py-20 px-6 bg-gradient-to-r from-[#1E1E1E] to-[#121212] border-y border-[#2A2A2A]">
        <div className="max-w-4xl mx-auto flex flex-col md:flex-row items-center gap-10">
          <div className="flex-1">
            <div className="inline-block px-3 py-1 bg-[#00D98E]/20 text-[#00D98E] rounded-md text-sm font-bold mb-4 uppercase tracking-widest">
              Ingyenes Ajándék
            </div>
            <h2 className="text-4xl font-bold mb-6">“Első 7 napi nyerő tippek csomagja”</h2>
            <p className="text-xl text-[#C0C0C0] mb-8">
              Azonnal kipróbálható tippek, statisztikai elemzéssel. Töltsd le a PDF-et és lásd a különbséget az első naptól!
            </p>
            <button
              onClick={() => handleWaitlistClick('lead_magnet')}
              className="px-8 py-4 bg-white text-[#1E1E1E] font-bold rounded-lg hover:bg-[#C0C0C0] transition-colors flex items-center gap-2"
            >
              PDF Letöltése Most
              <ArrowRight className="w-5 h-5" />
            </button>
          </div>
          <div className="w-full md:w-1/3 aspect-[3/4] bg-[#2A2A2A] rounded-xl border border-[#3A3A3A] flex items-center justify-center text-6xl shadow-2xl relative overflow-hidden group">
            <div className="absolute inset-0 bg-[#00D4FF]/5 group-hover:bg-transparent transition-colors"></div>
            📄
            <div className="absolute bottom-4 left-4 right-4 bg-black/60 backdrop-blur-md p-3 rounded-lg text-xs text-center border border-white/10">
              STABIL ROI STRATÉGIA PDF
            </div>
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
                className={`px-6 py-3 rounded-lg font-semibold transition-all ${activeTab === tab.id
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
                    { num: '1', title: 'Adatvezérelt elemzés', text: 'Saját rendszerünk több ezer meccset és 15+ statisztikai mutatót elemez valós időben (xG, forma, sérültek).' },
                    { num: '2', title: 'Statisztikai háttér', text: 'Minden tipphez mutatjuk a matematikai hátteret. Nem megérzésre, hanem valószínűségre építünk.' },
                    { num: '3', title: 'Havi riportálás', text: 'Rendszeres ROI riportokat és havi elemzéseket készítünk, hogy lásd a stabil fejlődésedet.' },
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
                    💬 <strong>Keress nyugodtan.</strong><br />
                    Amint tudok, Telegramon, illetve e-mailen is próbálok aktív lenni, keressetek nyugodtan: tipforgehq@gmail.com
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
              <h3 className="font-semibold mb-2">Béta árazás</h3>
              <p className="text-sm text-[#A9A9A9]">5 490 Ft/hó (7 990 Ft helyett)</p>
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
                Telegram
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
                <span className="text-[#A9A9A9]">Indítási kedvezményt (5 490 Ft helyett 7 990 Ft-ra fog nőni a béta után)</span>
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
            <div className="text-xs text-[#A9A9A9]">Várólista zárul december 8-án</div>
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