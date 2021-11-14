# Readme dataviz project

 -   Aoussedj Camélia - M1-app BD/IA
 - 14/11/2021
```
Le projet
```
 - Le but du projet est d'apprendre à développer une application de data visualisation de A à Z à l'aide de streamlit.
 - Pour cela, nous utiliserons les données mises à dispositions par le gouvernement sur ce drive https://drive.google.com/drive/folders/1R_9A9yPOzRQzMCyTDBEJms0u1ZCN7MbY
 
````
Le but de cette application
````
 - Le but de cette application est de permettre aux utilisateurs d'avoir une idée des prix de l'immobilier dans des départements donnés. 
 
````
Les fonctionnalités développées
````
- Concernant les dataset, l'utilisateur peut lui-même charger le jeu de données qu'il souhaite (tant que sa taille est <200MB). Sinon, les 5 dataset du site du gouvernement sont mis à sa disposition, il peut alors sélectionner l'année qu'il souhaite.
- L'utilisateur dispose d'une sidebar qui lui permet de sélectionner le type de bien, le nombre de pièces principales, la surface et la localisation. L'utilisateur peut également mettre une fourchette de prix.
 - Des KPI sont affichés ainsi qu'une map de la localisation des biens qui correspondent à ses attentes. Des histogrammes et d'autres fonctionnalités sont mises à disposition pour explorer les biens qui lui correspondent.
 - Enfin, une dernière fonctionnalité de Machine learning permet à l'utilisateur d'avoir une idée du prix du bien de ses rêves. Pour cela, il doit renseigner certains paramètre, toujours dans la sidebar.

 - BONUS : dans le fichier plotlyexpress_2020_extrait_Aoussedj.py , une fenêtre pop-up permet de visualiser les prix de l'immobilier et la localisation de ce dernier selon la date choisi via la slidebar en bas. De plus, s'il souhaite zoomer sur une fourchette de prix directement sur le graph, il le peut. Cela permet d'agrandir le graph sur cette zone et de montrer davantages de biens qui pourraient correspondre à ses besoins

