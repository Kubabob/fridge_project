## Used model

YOLO - you only look once, series of real-time object detection systems based on convolutional neural networks; requires only one forward propagation pass through the neural network to make predictions

we used segmentation (YOLO11n-seg) - literki to wielkości modeli i mnożniki wag

### food-recognition-2022/data_preparation.ipynb
dataset food-recognition-2022 zawiera prawie 500 klas obiektów; Training Set -> 76491 annotations -> 498 food classes; Validation Set -> 1830 annotations -> 498 food classes

na początek pipeline jak przerobić pobrany dataset (z plików COCO json) na dostępny dla modelu YOLO -> foldery images 640x640 i labels (jaki obiekt i opis jego umiejscowienia na obrazie) + dataset.yaml z nazwami wszystkich klas obiektów

label -> plik txt (class_id, x_centre,  y_centre,  width,  height)

## Model training

jest wstępnie wytrenowany na zbiorze danych COCO (Common Objects in Context), który zawiera 80 klas obiektów.

ładujemy nasze klasy, liczymy na karcie graficznej

możliwość zobaczenia architektury

Final learning rate fraction - Jest to ułamek (fraction) określający, do jakiej wartości końcowej (final LR) ma zostać zmniejszony początkowy learning rate (initial LR) w trakcie treningu

### Parametry
weight_decay: zapobiega przeuczeniu przez karanie dużych wag.


box: (Box loss gain) - Waga funkcji straty dla regresji bounding boxów.  Im wyższa wartość, tym większy nacisk na precyzję bboxów.

 
cls: (Class loss gain).Waga straty klasyfikacji (Cross-Entropy lub Focal Loss). Kontroluje wpływ błędów klasyfikacji na całkowitą stratę.


dfl: (Distribution Focal Loss gain). Waga straty DFL. Pomaga modelowi lepiej przewidywać rozkład granic bboxów.

### Model
zakończył się po 40 epokach, liczył się całą noc, nie jest najlepszym modelem, ale uznaliśmy, że już nie ma co robić dalej, pokażemy jego możliwości

tabela ze wszystkimi statystykami każdej klasy po walidacji

super 0.995:
- fruit salad
- croissant
- popcorn salted
- chorizo
- wine rose
  
słabo:
- chocolate 0.103
- basil 0.016
- nectarine 0.0815
- cottage cheese 0.0822
- ham 0.0148

### runs/segment/train
confusion_matrix: nie jest pewny prawie niczego, nawet jeśli coś jest czymś. możliwe że za dużo klas na jedno uczenie bądź za mało czasu na nauczenie się takiej ilości klas

F1: Jak wiele wykrytych bboxów to prawdziwe obiekty? (niska precyzja)
Precision = TP / (TP + FP)
Jak wiele prawdziwych obiektów zostało wykrytych? (niska czułość)
Recall = TP / (TP + FN)

F1-score:
F1 = 2 * (Precision * Recall) / (Precision + Recall)

tylko nieliczne klasy wybijają się ponad 0.6; a nam zależy na wartościach bliskich 1; model rzadko jest pewny swoich detekcji – większość przewidywań ma niskie confidence (np. < 0.1).

Możliwe przyczyny:
- Niedostateczna ilość danych treningowych (zwłaszcza dla niektórych klas). - u nas ok 150 obiektów na klasę
- Słaba jakość danych (np. nieprecyzyjne bounding boxy lub brak równowagi klas).
- Przeuczenie (overfitting) – model nie generalizuje na nowe dane.
- Zbyt krótki trening lub nieodpowiednie hiperparametry (np. zbyt niski box lub cls gain)


Precision: 
Osiąga 0.95 (dla wszystkich klas) przy maksymalnym progu pewności (1.0), model jest pewny na 100% (conf=1.0), jego detekcje są niemal idealne (mało fałszywych alarmów). Model jest bardzo konserwatywny – tylko detekcje z najwyższą pewnością są trafne.

Możliwe przyczyny:
- Zbyt wysoki próg pewności w treningu (np. model uczony na "idealnych" przykładach).
- Brak równowagi klas (model skupia się na łatwych klasach, ignorując trudne).
- Przeuczenie (overfitting) – model działa dobrze tylko na bardzo podobnych danych do treningowych.
- Słaba jakość danych walidacyjnych (np. brak adnotacji dla niektórych obiektów).

przykłady jak sobie poradził

### runs/segment/val7
confusion_matrix: jest jeszcze mniej pewny, zdażają się pomyłki (black-forest-tart i bread-wholemeal), nawet jeśli coś jest czymś. możliwe że za dużo klas na jedno uczenie bądź za mało czasu na nauczenie się takiej ilości klas

F1: Jak wiele wykrytych bboxów to prawdziwe obiekty? (niska precyzja)
Precision = TP / (TP + FP)
Jak wiele prawdziwych obiektów zostało wykrytych? (niska czułość)
Recall = TP / (TP + FN)

F1-score:
F1 = 2 * (Precision * Recall) / (Precision + Recall)

gorsze jeszcze, mniej potrafi wykrywać i to dobrze


Precision: 
dalej wysoka, ale tylko dla nielicznych
kiedy model niemal pewny jego detekcje są niemal idealne (mało fałszywych alarmów). Model jest bardzo konserwatywny – tylko detekcje z najwyższą pewnością są trafne.

Recall: bardzo nisko czuły na wykrywanie obiektów

przykłady jak sobie poradził, np lemoniada wykryta jako woda guess why

### nasz przykład
food-recognition-2022/evaluation_results/result-white-bread-cheese 1 i 2
spoko mu poszło, ale wiemy że z chlebem mu dobrze szło

przykłady z walidacji - słabe

## food-recognition-2022/results_visualization

statystyki dla wszystkich epok, wizualizacja

skoki w validation losses classification - dziwne mijesce względem lokalnego minimum, mało optymalne wagi; reszta w miarę optymalnie;
- Niestabilne batch'e walidacyjne; Niektóre epoki trafiają na "trudniejsze" batch'e walidacyjne (np. z rzadkimi klasami lub zakłóceniami).
- Konflikt gradientów: Gdy model jednocześnie uczy się detekcji (Box Loss) i klasyfikacji (Classification Loss), gradienty mogą się "prześcigać"

ogólnie się uczy

-Box Loss (strata dla bounding boxów) i Segmentation Loss (maski segmentacji) są najwyższe, co sugeruje, że model ma największe trudności z precyzyjną lokalizacją obiektów i ich kształtami.
-Classification Loss jest stosunkowo niska – model dobrze rozpoznaje klasy.
-DFL Loss (Distribution Focal Loss) również maleje, ale może wymagać dalszej optymalizacji.

Precision (nie umie usuwać fałszywych detekcji) x Recall (wiele prawdziwych obiektów pomija): W późnych epochach Precision > Recall → model jest bardziej ostrożny (woląc pominąć obiekt niż dać FP).; 

mAP: twórcy YOLO doszli do 0.3; my mamy 0.2<; nie najgorzej, widac tredn, który pozwala sugerować, że też do tylu byśmy doszli

aMP (ang. average Mask Precision) to kluczowy wskaźnik wydajności modeli segmentacji instancji (instance segmentation), który mierzy średnią precyzję masek dla wszystkich klas obiektów. Jest to rozszerzenie standardowej metryki mAP (mean Average Precision) o ocenę jakości segmentacji pikseli.

learning rate spada, bo skończył się warmup


małe różnice train-validation: dobrze generalizuje, ale jeszcze wiele musi się nauczyć