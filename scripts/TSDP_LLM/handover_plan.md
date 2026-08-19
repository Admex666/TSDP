# 1️⃣ Product Brief

* **One-sentence pitch:** Egy Football Knowledge Base AI, amely futballkönyvekből lokálisan felépített tudásbázis alapján természetes nyelvű kérdésekre válaszol, releváns források megjelölésével.
* **Problem statement:** A futballszakirodalomban található információk nagy mennyiségű, nehezen kereshető anyagban vannak szétszórva, ezért nehéz gyorsan releváns tudást megtalálni és összekapcsolni.
* **Why now:** N/A
* **Target audience:** Futballszakmai és elemzői felhasználók, elsősorban a projekt tulajdonosa.

# 2️⃣ Functional Spec

* **Must-have features:**

  * Futballkönyvek feldolgozása egy kijelölt könyvtárból.
  * PDF-ekből szöveg kinyerése.
  * A könyvek szövegének chunkokra bontása.
  * Lokális embeddingek készítése.
  * A chunkok és metaadataik tárolása vektordatabázisban.
  * Természetes nyelvű kérdések alapján releváns könyvrészletek keresése.
  * RAG-alapú válaszgenerálás.
  * Két LLM-backend támogatása:

    * lokálisan futó LLM Ollama segítségével;
    * ingyenes API-n keresztül elérhető LLM OpenRouter segítségével.
  * A két LLM-verzió összehasonlíthatósága ugyanazon tudásbázison.
  * A válaszokhoz releváns könyv- és oldalszám-források megjelenítése.
  * Egyszerű chat UI a tudásbázissal való interakcióhoz.

* **Nice-to-have features:**

  * N/A

* **Out of scope:**

  * Futballadatok és event/tracking adatok integrációja az első verziókban.
  * Saját LLM modell fejlesztése vagy újratanítása.
  * Fizetős LLM API használata.
  * Saját ChatGPT-szintű általános célú modell építése.
  * Kubernetes, Docker Swarm vagy hasonló komplex infrastruktúra használata.
  * A V3 utáni funkciók.

* **User flows (step-by-step):**

  1. A felhasználó elhelyezi a jogszerűen hozzáférhető futballkönyveket egy kijelölt könyvtárban.
  2. A rendszer beolvassa a PDF-eket.
  3. A rendszer kinyeri a szöveget.
  4. A rendszer chunkokra bontja a szöveget.
  5. A rendszer lokális embeddingeket készít a chunkokhoz.
  6. A rendszer eltárolja a chunkokat és metaadataikat a vektordatabázisban.
  7. A felhasználó természetes nyelven kérdést tesz fel.
  8. A rendszer embeddinget készít a kérdésből.
  9. A rendszer megkeresi a releváns könyvrészleteket.
  10. A rendszer a releváns részleteket átadja a kiválasztott LLM-backendnek.
  11. A kiválasztott LLM választ generál.
  12. A rendszer megjeleníti a választ és a releváns forrásokat.
  13. A felhasználó szükség esetén ugyanazt a kérdést a másik LLM-backenddel is lefuttatja.

# 3️⃣ Technical Constraints

* **Preferred stack:**

  * Python.
  * PyMuPDF PDF-feldolgozáshoz.
  * Saját Python-alapú chunking.
  * Lokális embedding modell.
  * Qdrant vektordatabázisként.
  * Ollama lokális LLM-futtatáshoz.
  * Open-weight, quantizált lokális modell.
  * OpenRouter free API LLM-backendként. Több API-kulcs rotálása szolgáltatói limitek megkerülésére.
  * Streamlit a chat UI-hoz.
  * A két LLM-backend közös interfészen keresztül legyen elérhető.

* **Must avoid:**

  * Fizetős LLM API-k.
  * Saját LLM újratanítása.
  * LangChain használata az első projektverzióban.
  * Feleslegesen komplex infrastruktúra.
  * A teljes könyvtár külső LLM API-ba történő továbbítása; az API-backend csak a retrieval során kiválasztott releváns kontextust kapja.

* **Scalability expectations:**

  * Az első verzió egy futballkönyvekből álló tudásbázis kezelésére készül.
  * A rendszernek később lehetőséget kell biztosítania további futballkönyvek és futballadatok integrációjára.
  * A kezdeti hardver:

    * i5-10400F CPU.
    * 16 GB RAM.
    * GeForce GTX 1650 Super, 4 GB VRAM.
  * A lokális LLM-nél a hardver miatt kisebb, körülbelül 7–8B paraméteres quantizált modellek használata az irány.
  * Az API-backend esetében az ingyenes szolgáltatói limiteket figyelembe kell venni.

* **Security/privacy constraints:**

  * A rendszer lokálisan fusson, amennyiben a local LLM-backendet használjuk.
  * A könyvek és a tudásbázis alapvetően lokálisan legyenek tárolva.
  * Az OpenRouter-backend használatakor csak a retrieval által kiválasztott releváns könyvrészletek kerüljenek külső szolgáltatóhoz.
  * Az API-backend használatakor figyelembe kell venni az adott free modell adatkezelési feltételeit. Az OpenRouter free modelljei között vannak olyanok, amelyeknél a provider a promptokat és outputokat naplózhatja és fejlesztésre használhatja.
  * Csak jogszerűen hozzáférhető könyvek kerüljenek a rendszerbe.

# 4️⃣ Decision log

* **We explicitly decided:**

  * A projekt első célja egy Football Knowledge Base AI.
  * Az első három fejlesztési fázis:

    1. Knowledge ingestion.
    2. RAG + LLM.
    3. Egyszerű chat UI.
  * Az első verziókban nem lesznek futballadatok integrálva.
  * A rendszer lokálisan tárolja a tudásbázist.
  * Fizetős LLM API helyett két alternatív LLM-backendet tesztelünk.
  * Az egyik backend teljesen lokális LLM lesz Ollamával.
  * A másik backend ingyenes API-n keresztül működő LLM lesz OpenRouterrel. API-kulcsokat rotálhatunk a szolgáltatói limitek megkerülésére.
  * A két backend ugyanazt a RAG pipeline-t használja.
  * Qdrant lesz a vektordatabázis.
  * Python lesz az alap technológia.
  * PyMuPDF lesz a PDF-feldolgozás alapja.
  * Saját Python chunkingot használunk.
  * Lokális embedding modellt használunk.
  * Streamlit lesz a V3 chat UI technológiája.
  * Az első verzióban nem használunk LangChaint.
  * A felhasználó hardvere:

    * i5-10400F.
    * 16 GB RAM.
    * GeForce GTX 1650 Super, 4 GB VRAM.
  * A rendszernek forrásmegjelöléseket kell adnia, lehetőleg könyv- és oldalszámmal.
  * A local és OpenRouter LLM-backendeket össze kell tudni hasonlítani ugyanazon kérdések és ugyanazon retrieval context alapján.

* **We explicitly rejected:**

  * Fizetős LLM API használata.
  * Saját LLM modell újratanítása.
  * Nagy, saját GPU-infrastruktúrát igénylő modell használata az első verzióban.
  * LangChain használata az első projektverzióban.
  * Feleslegesen komplex infrastruktúra.
  * Futballadatok integrálása az első három fejlesztési fázisba.

* **Still undecided:**

  * A konkrét lokális embedding modell.
  * A konkrét lokális LLM.
  * A konkrét OpenRouter free LLM.
  * Az OpenRouter free modellek közül egy konkrét modell rögzítése vagy az `openrouter/free` automatikus modellválasztás használata. Az OpenRouter jelenleg mindkét lehetőséget biztosítja.
