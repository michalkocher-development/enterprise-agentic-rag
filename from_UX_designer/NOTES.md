# Workbench UI — dokumentacja prototypu

Prototyp: `agentic-rag-workbench.html`. Jeden plik, otwiera się dwuklikiem, nie wymaga backendu.
Strumień SSE jest zamockowany, więc możesz obejrzeć całą choreografię 20-sekundowego przebiegu,
zanim podepniesz FastAPI.

Do obejrzenia w pierwszej kolejności: przycisk **„Ile NVIDIA zarobiła na kartach do grania?"**.
To jedyny scenariusz, który przechodzi przez pętlę autokorekty.

---

## 1. Co się zmieniło i dlaczego

### Typografia niesie hierarchię

Poprzednio wszystko było monospace: nagłówki, opisy, przyciski, placeholdery. Monospace w interfejsie
sygnalizuje „to są dane maszynowe", więc kiedy jest wszędzie, przestaje sygnalizować cokolwiek —
i cały ekran czyta się jak zrzut z terminala.

Teraz Inter niesie chrome, a JetBrains Mono jest zarezerwowany wyłącznie dla wartości, które faktycznie
pochodzą z maszyny: scoring cross-encodera, VRAM, czasy w ms, nazwy plików, identyfikator sesji, liczniki.
Ta jedna zmiana odpowiada za większość różnicy w odbiorze.

### Warstwy zamiast ramek

Zamiast jednolitego `1px solid` na płaskiej czerni jest trójstopniowa elewacja:

| Token | Wartość | Zastosowanie |
| --- | --- | --- |
| `bg` | `#08090A` | tło strony |
| `panel` | `#0D0F10` | panele, karty |
| `raised` | `#14171A` | węzły grafu, aktywne przyciski |
| `line` | `rgba(255,255,255,.08)` | domyślna ramka |
| `line2` | `rgba(255,255,255,.14)` | hover, akcenty strukturalne |

Ramka nadal istnieje, ale jest ledwie widoczna — podział niesie kontrast tła, nie linia.

### Jeden kolor akcentu, jedno znaczenie

`--live: #6C8CFF` oznacza **wyłącznie** „ten węzeł pracuje w tej chwili". Nic innego w interfejsie
nie ma prawa go użyć. Dzięki temu przez całe 20 sekund oko wie, gdzie patrzeć, bez czytania.

Kolory semantyczne są rozdzielne i mają stałe znaczenie: `#3FBF8F` ugruntowane / trafne,
`#E0A44A` pętla i odrzucenia, `#E2564D` halucynacja.

### Cały ekran mieści się w 100vh

Rekruter nie może scrollować, żeby zobaczyć graf i odpowiedź naraz — scrollowanie zrywa związek
między „agent właśnie wykonuje węzeł Rerank" a „oto co ten węzeł zrobił". Layout jest więc sztywny:

```
header            48 px   stała
pas grafu        ~210 px  stała, proporcja SVG 7,3:1
dwie kolumny     flex-1   scrolluje się wyłącznie ich WNĘTRZE
belka wejścia     58 px   stała
```

`body` ma `lg:h-screen lg:overflow-hidden lg:flex lg:flex-col`, a każda kolumna
`flex flex-col min-h-0` z wewnętrznym `overflow-y-auto`. Kluczowe jest `min-h-0` — bez niego
element flex nie skurczy się poniżej rozmiaru treści i scroll wycieknie na całą stronę.

Przy 1440×780 (typowy MacBook 13") kolumny dostają ~440 px, przy 1920×1080 ~700 px.
Poniżej breakpointu `lg` układ wraca do zwykłego scrollowania — na telefonie sztywne 100vh
jest gorsze niż scroll.

Lista dziesięciu kandydatów jest przez to jednowierszowa (40 px), a nie dwuwierszowa: cała
dziesiątka mieści się w kolumnie inspektora bez scrollowania, więc animacja przetasowania
jest widoczna w całości. To był realny koszt — podgląd treści fragmentu jest teraz ucięty
do jednej linii.

### Graf jest grafem

Pełnoszerokościowy pas pod nagłówkiem, osiem węzłów, warunkowe krawędzie, dwie pętle.
Węzły stoją w jednym rzędzie, a obie pętle są wyprowadzone poza ten rząd: autokorekta nad nim,
regeneracja i pominięcie RAG pod nim. Dzięki temu `viewBox` ma proporcję 1400×192 zamiast
1120×292 i pas zajmuje ~190 px zamiast ~300, nie tracąc ani jednej krawędzi.
Krawędź, którą właśnie płynie sterowanie, ma animowany `stroke-dashoffset`; krawędź już przebyta
zostaje trwale podświetlona; krawędzie nieużyte zostają wygaszone. Po zakończeniu przebiegu
sam kształt podświetlenia opowiada, jaką drogą poszedł agent.

Każdy węzeł ma cztery stany (`idle` / `active` / `done` / `skipped`) sterowane atrybutem
`data-state` na `<g class="gnode">`. Cała reszta jest w CSS.

### Zakładki wycięte

Poprzednie `1. RETRIEVER / 2. RERANK / 3. GRADER / 4. AUTOKOREKTA` wymagały ręcznego klikania
w trakcie przebiegu, żeby cokolwiek zobaczyć — czyli ukrywały dokładnie tę historię, którą chcesz
opowiedzieć. Zastąpione pionową osią czasu, która dopisuje sekcje w miarę przebiegu i autoscrolluje.
Każda sekcja ma własny licznik czasu, który zamienia się w wynik po zamknięciu węzła.

### Reranking jako moment kulminacyjny

Cztery sekundy inferencji na GPU to najdłuższy pojedynczy element poza generacją, a jednocześnie
najmniej zrozumiały. Prototyp pokazuje: dziesięć kandydatów w kolejności z wektorówki → przetasowanie
z animacją do kolejności cross-encodera → wagi wjeżdżają jako paski → sześć odrzuconych gaśnie →
lista zwija się do czterech. Nikt nie potrzebuje wyjaśnienia, po co jest reranker, jeśli to zobaczy.

Technicznie: elementy są `position:absolute` z `transform: translateY(index * 58px)` i przejściem
na `transform`. Kolejność zmienia się przez przeliczenie transformów, nie przez zmianę DOM,
więc animacja jest płynna i nie ma migotania.

### Cytowania są dwukierunkowe

Najechanie na cytowanie w odpowiedzi podświetla ten sam fragment we wszystkich sekcjach osi czasu
i przewija go do widoku. Spina warstwę „oto odpowiedź" z warstwą „oto dowód". Działa na `data-chunk`
jako wspólnym identyfikatorze — dlatego identyfikatory fragmentów muszą być stabilne w obrębie przebiegu
(patrz rekomendacja 5).

### Przykładowe zapytania mają role

Trzy przyciski, każdy pokazuje inną ścieżkę przez graf: prostą, z pętlą autokorekty, i z pominięciem RAG.
Ten drugi jest podpisany „wymusza autokorektę" i wyróżniony bursztynową ramką, bo to jest dowód,
że budujesz agenta, a nie pipeline.

---

## 2. Struktura kodu

```
<head>
  konfiguracja Tailwind (paleta jako nazwane kolory)
  <style>  — tylko to, czego Tailwind nie robi:
             animacje krawędzi, stany węzłów, reordering listy,
             kursor streamingu, prefers-reduced-motion
<body>            lg:h-screen lg:overflow-hidden lg:flex lg:flex-col
  header          — GPU, tempo, przełącznik PL/EN            [stała wysokość]
  section .graph  — SVG: krawędzie, etykiety warunków, węzły [stała wysokość]
  main            — grid: [odpowiedź] | [oś czasu]           [flex-1, min-h-0]
  footer          — przykłady + input + wyślij               [stała wysokość]
<script>
  I18N            — słownik pl/en, funkcja t(), applyLang()
  DOCS / POOL_*   — mockowe fragmenty
  SCEN            — trzy scenariusze jako listy kroków
  render*()       — jeden renderer na typ ładunku węzła
  run()           — sterownik przebiegu
```

Sterownik `run()` jest napisany tak, żeby podmiana mocka na prawdziwy `EventSource` była
lokalną zmianą: pętla `for (const st of sc.steps)` zamienia się w handler `onmessage`,
a wszystkie `render*()` zostają bez zmian. Nie ma innych miejsc, które trzeba ruszać.

### i18n

Zero stringów w markupie. Chrome ma `data-i18n="klucz"`, dane scenariuszy mają warianty
`{pl: …, en: …}`. `applyLang()` przechodzi po wszystkich `[data-i18n]` i podmienia `textContent`.
Dopisanie trzeciego języka to jeden obiekt w `I18N` i jeden przycisk.

### Dostępność

Ograniczenia ruchu (`prefers-reduced-motion`) wyłączają wszystkie animacje. Fokus klawiatury jest
widoczny. SVG ma `role="img"` i `<desc>`. Kolor nigdy nie jest jedynym nośnikiem znaczenia —
werdykty mają etykiety słowne, cytowania numery.

### Kontrolka tempa

`1× / 2× / 4×` skaluje wszystkie opóźnienia. Zostawiłem ją także dla wersji produkcyjnej,
podpisaną „tempo demo": ktoś, kto ogląda drugi przebieg, nie chce czekać kolejnych 20 sekund.
Domyślnie zawsze `1×` — pierwsze wrażenie ma być pełne.

---

## 3. Rekomendacje dla backendu

Uszeregowane od największego wpływu na UX.

### 3.1 Zdarzenie `node_start` — konieczne

Dziś `step` leci, gdy węzeł **skończy**. Przez cztery sekundy inferencji na GPU frontend nie ma
żadnej informacji, że rerank się zaczął. Nie da się na tym zbudować uczciwego stanu „active"
ani licznika czasu lecącego na żywo — pozostaje zgadywanie z faktu, że poprzedni węzeł zamknął,
co rozjeżdża się przy każdym opóźnieniu sieci.

```json
event: node_start
data: {"node": "local_rerank", "edge_from": "retrieve", "iteration": 1}
```

Pole `edge_from` pozwala frontendowi animować konkretną krawędź, zamiast wyprowadzać ją
z topologii zaszytej w JS. Przy pętlach to nie jest kosmetyka — krawędź `grade → rewrite`
i `grade → generate` wychodzą z tego samego węzła i tylko backend wie, którą wybrał router.

W LangGraph najprościej wpiąć to w `astream_events` i filtrować `on_chain_start` po nazwach węzłów.

### 3.2 Streaming tokenów w `generate` — konieczne

Najdłuższy pojedynczy węzeł. Jeśli zostanie czarną skrzynką, dostajesz martwe pole dokładnie
w kulminacyjnym momencie przebiegu. Reszta węzłów niech zostanie jak jest — one mają dyskretne
wyniki, nie tekst.

```json
event: token
data: {"node": "generate", "delta": "wyniósł "}
```

`astream_events` z `on_chat_model_stream` daje to bez przebudowy grafu.

### 3.3 Stabilne identyfikatory fragmentów przez cały przebieg — konieczne

Prototyp spina cytowanie z wierszem w rerankingu i z werdyktem gradera po wspólnym `chunk_id`.
Jeśli backend wysyła w `retrieve` indeks pozycyjny, w `grade_documents` nazwę pliku, a w `generate`
sam tekst cytatu, to spięcie jest niemożliwe i cała warstwa dowodowa się rozpada.

Nadaj każdemu parent chunkowi identyfikator w momencie ingestii i przenoś go przez wszystkie
zdarzenia bez zmian:

```json
{"chunk_id": "nvidia_q3_fy25_10q#p14", "parent_id": "...", "filename": "...", "is_table": true}
```

Przy pętli autokorekty ten sam fragment może wrócić w drugim przebiegu — wtedy identyfikator
się powtarza i to jest pożądane, bo frontend może pokazać „ten fragment już był odrzucony".

### 3.4 Wagi dla wszystkich kandydatów, nie tylko top-4

Animacja przetasowania listy wymaga scoringu dla całej dziesiątki. Dziś wysyłasz cztery
wyselekcjonowane, więc frontend nie ma jak pokazać, **co** reranker odrzucił i jak bardzo.
Sześć dodatkowych floatów w ładunku, a to jest najlepsze cztery sekundy w całym demie.

```json
event: step
data: {"node": "local_rerank",
       "ranked": [{"chunk_id": "...", "score": 0.94, "kept": true}, …],
       "vram_mb": 412, "duration_ms": 3910}
```

### 3.5 Licznik iteracji pętli w stanie grafu

`GraphState` powinien nieść `rewrite_count` i `regenerate_count` razem z limitami. Frontend pokazuje
wtedy `2/3` na krawędzi pętli, co jest jednym z najmocniejszych sygnałów, że to maszyna stanów,
a nie sekwencja. Przy okazji ratuje cię przed nieskończoną pętlą, jeśli limit jest dziś tylko
w kodzie warunku, a nie w stanie.

### 3.6 `duration_ms` w każdym ładunku, `total_time_ms` w `complete`

Frontend może mierzyć czas sam, ale zmierzy wtedy czas z narzutem sieci i renderowania.
Jeśli chcesz pokazywać liczby jako telemetrię — a chcesz, to jest projekt LLMOps — muszą pochodzić
z serwera. Zwłaszcza `local_rerank`, gdzie chodzi o inferencję na GPU, a nie o round-trip HTTP.

### 3.7 Keepalive podczas długich węzłów

Cztery sekundy ciszy na strumieniu SSE to dużo. `X-Accel-Buffering: no` masz, ale przy hostingu
za Cloudflare, Renderem czy nginxem z domyślną konfiguracją długa cisza kończy się zerwaniem
połączenia. Wysyłaj komentarz co sekundę:

```
: keepalive
```

To jedna linia, ale bez niej demo umiera na cudzej infrastrukturze, a nie na twoim laptopie.

### 3.8 Zdarzenie `error` z nazwą węzła

Jeśli Gemini zwróci 429 albo skończy się darmowy limit, dziś strumień prawdopodobnie po prostu
się urywa i UI zostaje z wiecznie kręcącym się węzłem. Osoba oglądająca portfolio nie wie,
że to limit API — widzi zepsutą aplikację.

```json
event: error
data: {"node": "generate", "code": "rate_limit", "retry_after_s": 34}
```

Frontend ustawia wtedy węzeł w stan `error` i pokazuje konkretny komunikat.

### 3.9 Odtwarzanie zapisanych przebiegów — rekomendacja produktowa

To jest rzecz, która najbardziej zwiększy skuteczność portfolio na jednostkę pracy.

Twoje demo zależy od klucza Gemini, darmowego limitu i od tego, czy laptop z RTX 2050 jest włączony.
Hiring manager otwiera link o 23:00 w niedzielę. Jeśli którykolwiek z tych warunków nie jest spełniony,
widzi martwą stronę i nie wraca.

Zapisuj każdy przebieg jako listę zdarzeń w JSON i dodaj endpoint `/api/v1/replay/{run_id}`,
który odtwarza je z oryginalnymi opóźnieniami. Trzy przykładowe zapytania serwuj domyślnie z replaya,
a prawdziwą inferencję odpalaj dopiero dla własnych pytań użytkownika — z czytelnym oznaczeniem,
które jest które. Demo działa wtedy zawsze, koszt API spada do zera, a osoba techniczna docenia,
że pomyślałeś o determinizmie.

Format zdarzeń w replayu jest identyczny jak w strumieniu na żywo, więc frontend nie musi
o tym wiedzieć. To ~40 linii w FastAPI.

### 3.10 Parametr języka w żądaniu

Skoro interfejs ma przełącznik PL/EN, odpowiedź LLM też musi za nim iść. Dodaj `lang` do payloadu
`/api/v1/chat/stream` i wstrzyknij do promptu w `generate`. Uzasadnienia gradera i przepisane
zapytanie też — inaczej dostajesz angielski interfejs z polskimi werdyktami.

### 3.11 Rozgrzewka modelu przy starcie

Jeśli `bge-reranker-base` ładuje się leniwie przy pierwszym zapytaniu, pierwszy przebieg ma
kilkanaście sekund w węźle rerank i wygląda na zawieszony. Załaduj model i wykonaj jedną
pustą inferencję w `lifespan` FastAPI, a w UI pokaż stan gotowości GPU w nagłówku
(prototyp ma tam kropkę — dziś zawsze zieloną, docelowo sterowaną z `/health`).

---

## 4. Podpięcie prawdziwego strumienia

W `run()` zamień pętlę po `sc.steps` na:

```js
const es = new EventSource(`/api/v1/chat/stream?q=${encodeURIComponent(q)}&lang=${LANG}`);

es.addEventListener('session',   e => { $('#sessionId').textContent = JSON.parse(e.data).thread_id; });
es.addEventListener('node_start',e => onNodeStart(JSON.parse(e.data)));
es.addEventListener('step',      e => onStep(JSON.parse(e.data)));
es.addEventListener('token',     e => appendToken(JSON.parse(e.data).delta));
es.addEventListener('error',     e => onError(JSON.parse(e.data)));
es.addEventListener('complete',  e => { onComplete(JSON.parse(e.data)); es.close(); });
```

`onNodeStart` robi to, co dziś robią `setNode(node,'active')` + `flowEdge(edge,true)` + `tlSection(node)`.
`onStep` wybiera renderer po `data.node` — to jest ten sam `switch`, który już jest w pętli.
Wszystkie funkcje `render*()` przyjmują już ładunek w docelowym kształcie, więc nie wymagają zmian.

Jedyna rzecz do dopisania: obsługa rozłączenia. `es.onerror` przy zerwanym połączeniu powinno
zostawić graf w stanie, w jakim był, i pokazać możliwość wznowienia — a nie czyścić ekran.

---

## 5. Czego prototyp jeszcze nie ma

Świadome pominięcia, do dołożenia przy wdrożeniu:

- **Historia konwersacji.** Prototyp pokazuje jeden przebieg. Docelowo poprzednie pary
  pytanie–odpowiedź powinny zwijać się do jednej linijki nad aktualną, z możliwością rozwinięcia
  ich przebiegu. Graf zawsze pokazuje ostatni.
- **Upload dokumentu.** Potok ingestii ma własną choreografię (normalizacja → OCR → chunking →
  embeddingi) i zasługuje na osobny, mniejszy widok postępu. Nie mieszać z grafem zapytania.
- **Podgląd fragmentu.** Kliknięcie w wiersz powinno otwierać panel z pełnym parent chunkiem —
  szczególnie dla tabel, bo to jest cały sens Parent-Document Retrieval i dziś nigdzie tego nie widać.
- **Widok po zakończeniu.** Po `complete` warto pozwolić przełączyć oś czasu w tryb porównawczy
  dla przebiegów z pętlą: pierwszy zestaw kandydatów obok drugiego, żeby było widać, co dała korekta.
- **Podgląd treści fragmentu.** Wiersze listy są jednowierszowe, żeby cała dziesiątka mieściła się
  bez scrollowania. Pełny podgląd powinien wjeżdżać na hover w tooltipie albo w panelu bocznym.
- **Responsywność poniżej 1024 px.** Poniżej `lg` layout wraca do zwykłego scrollowania, a graf
  o proporcjach 1400×192 robi się bardzo niski. Na wąskich ekranach warto przełączyć go
  na układ pionowy albo zwinąć do paska postępu z nazwą aktywnego węzła.
- **Bardzo niskie okna (<640 px wysokości).** Przy takiej wysokości kolumny dostają ~200 px
  i robi się ciasno. Warto dodać `@media (max-height: 640px)`, które zwija pas grafu
  do samego rzędu węzłów bez etykiet krawędzi.
