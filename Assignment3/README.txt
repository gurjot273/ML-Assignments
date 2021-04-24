Instructions for running source codes:
The current working directory must be src
cd src 
For task A, run python probA.py
This creates tf-idf.csv inside data/clusters folder, which is the dataset generated during part A
For task B, run python probB.py
This creates agglomerative.txt
For task C, run python probC.py
This creates kmeans.txt
For task D, run python probD.py
This creates agglomerative_reduced.txt and kmeans_reduced.txt
For task E, run python probE.py
This prints the NMI scores for the four clusters

Scores:

NMI for cluster corresponding to agglomerative.txt: 0.024531706446442935
NMI for cluster corresponding to kmeans.txt: 0.5159780308432939
NMI for cluster corresponding to agglomerative_reduced.txt: 0.026994623860186414
NMI for cluster corresponding to kmeans_reduced.txt: 0.6953471454666308