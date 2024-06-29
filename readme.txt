Importante:
-> rrt_.py: aici sunt implementate și testate (cu toate scenariile de test înafară de cel de timp real, care este în main.py) toate versiunile de algoritmi RRT;
-> genetic_.py: implementare algoritm genetic;
-> main.py: s-au implementat funcțiile de citire din fișiere, construcția de hartă, toate operațiile adiacente pentru vizualizare, apelare funcții și vizualizare soluții;
-> astar.py: implementarea algoritmului A*;
-> mapstate_.py: conține enum-ul cu stările posibile și valorile lor de pe hartă;

Adiacente:
-> util_.py: implementare algoritm de linie bresenham;
-> test_plot_state.py: folosit pentru vizualizarea a diferite scenarii pe parcursul dezvoltării;
-> rrt_test.py: fișier folosit la început pentru familiarizarea cu algoritmul RRT;
-> rrt_star.py: fișier gol pentru că am ajuns la concluzia că este mai comod ca metodele să fie dezvoltate în același fișier pentru a avea acces la resurse comune, acum după finalizare o asemenea diviziune ar putea ajuta pentru modularitatea de termen lung;
-> random_test.py: folosit pentru a testa diferite intrări și ieșiri pentru funcții din python;
-> nn_test.py: folosit pentru a mă familiariza cu lucrarea cu rețele neuronale în python;
-> merge_excels.py: încercare pentru a centraliza rezultate, momentan nefuncțional;
-> genetic_nn.py: tentativa de a combina abordarea cu rețele neuronale cu algoritmi genetici;
-> DBSCAN_test.py: pentru a putea vizualiza rezultatele filtrării DBSCAN într-un mediu complet controlat;

Fișiere excel:
-> Cele din result*/ : rezultatele de la ultima testare realizată, media lor;
-> result* : rezultate parțiale din timpul dezvoltării;
-> model_params.xlsx: parametrii obținuți pentru rețeaua neuronală pe parcursul testării;
-> *.bag: fișiere obținute de pe lidare;
-> *.txt: conversia fișierelor .bag pentru a se putea lucra pe ele;
	-> Excepții:
		-> run_all.txt: fișier de configurare ROS de pe robot;
		-> readme.txt: fișierul acesta în sine;
