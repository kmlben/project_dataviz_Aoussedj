# This app is for educational purpose only. Insights gained are not financial advice. Use at your own risk!
from altair.vegalite.v4.schema.channels import Latitude
import streamlit as st
from streamlit_metrics import metric_row
from PIL import Image
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import time

#------------------------------------#
# Titre de la page et disposition des modules sur la page
#------------------------------------#
st.set_page_config(layout="wide")

# Titre
#------------------------------------#
image = Image.open('logo_efrei_finances_publiques.png')

st.image(image, width = 600)

st.title('Transactions immobilières en France')
st.markdown("""
Cette application permet de visualiser des jeux de données sur les transactions immobilières en France. Utilisez les Dataset proposés ou téléchargez le vôtre !

""")


# A propos
#------------------------------------#

expander_bar = st.beta_expander("À propos")
expander_bar.markdown("""
* **Module :** Data Visualization
* **Source des dataset de valeurs foncières :** [Site du gouvernement](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/#_).
* **Application développée par :** [Camélia Aoussedj] (https://www.linkedin.com/in/camelia-aoussedj/).
""")


# Division de la page
#------------------------------------#

col1 = st.sidebar
col2, col3 = st.beta_columns((2,1))

#-----------------------------------#
# Création de la sidebar et chargement des données
#-----------------------------------#

## Sidebar
st.sidebar.header('Chargement des données')

## Possibilité d'upload ses propres données
expander_bar2 = st.sidebar.beta_expander("Chargez votre propre jeu de données")
expander_bar2.markdown("""
* Chaque ligne doit être unique
* Les colonnes suivantes doivent obligatoirement figurer dans votre dataset : `code_departement, type_local, nombre_pieces_principales, valeur_fonciere, surface_terrain, longitude, latitude`
"""
                       )
uploaded_file = expander_bar2.file_uploader("Chargez vos données", type=["csv"])


## Possibilité de choisir parmi les 5 datasets mis à disposition, selon l'année

expander_bar1 = st.sidebar.beta_expander("Sélectionnez un dataset préenregistré")
with expander_bar1:
    datasetSelectionne = st.radio("Valeurs foncières de l'année :", ('2020','2019','2018','2017','2016'))

## Utilisation d'un cache

@st.cache
def load_data(nrows):
    data = pd.read_csv('full_2020_extrait.csv', nrows=nrows)
    return data

## Condition de création/utilisation des datasets

if uploaded_file is not None:
    # Cas 1 (l'utilisateur charge ses proposes data) : doit être .csv de moins de 200MB
    df = pd.read_csv(uploaded_file)
    uploaded_file.seek(0)
else:
    # Cas 2 (l'utilisateur ne charge pas ses proposes data) : le premier .csv de la BDD sera lu par défaut
    df = pd.read_csv('full_2020_extrait.csv')

## Il suffit d'enlever les commentaire ci-dessous pour avoir accès à cette fonctionnalité de la sidebar
## mais avec streamlit share impossible de charger les fichiers. Je me contente donc du fichier full_2020_extrait.csv
## qui est un extrait de full_2020.csv

#    if datasetSelectionne == '2020':
#        df = pd.read_csv('full_2020.csv')
#    elif datasetSelectionne == '2019':
#        df = pd.read_csv('full_2019.csv')
#    elif datasetSelectionne == '2018':
#        df = pd.read_csv('full_2018.csv')
#    elif datasetSelectionne == '2017':
#        df = pd.read_csv('full_2017.csv')
#    elif datasetSelectionne == '2016':
#        df = pd.read_csv('full_2016.csv')


### Dictionnaire des departements de France qui nous servirons à afficher le Nom du département 
### au lieu du de département (car seul le code département est présent dans les Dataset). Cela favorise l'ergonomie de l'application 

dictDepartements = {
    '1': 'Ain', 
    '2': 'Aisne', 
    '3': 'Allier', 
    '4': 'Alpes-de-Haute-Provence', 
    '5': 'Hautes-Alpes',
    '6': 'Alpes-Maritimes', 
    '7': 'Ardèche', 
    '8': 'Ardennes', 
    '9': 'Ariège', 
    '10': 'Aube', 
    '11': 'Aude',
    '12': 'Aveyron', 
    '13': 'Bouches-du-Rhône', 
    '14': 'Calvados', 
    '15': 'Cantal', 
    '16': 'Charente',
    '17': 'Charente-Maritime', 
    '18': 'Cher', 
    '19': 'Corrèze', 
    '2A': 'Corse-du-Sud', 
    '2B': 'Haute-Corse',
    '21': 'Côte-d\'Or', 
    '22': 'Côtes-d\'Armor', 
    '23': 'Creuse', 
    '24': 'Dordogne', 
    '25': 'Doubs', 
    '26': 'Drôme',
    '27': 'Eure', 
    '28': 'Eure-et-Loir', 
    '29': 'Finistère', 
    '30': 'Gard', 
    '31': 'Haute-Garonne', 
    '32': 'Gers',
    '33': 'Gironde', 
    '34': 'Hérault', 
    '35': 'Ille-et-Vilaine', 
    '36': 'Indre', 
    '37': 'Indre-et-Loire',
    '38': 'Isère', 
    '39': 'Jura', 
    '40': 'Landes', 
    '41': 'Loir-et-Cher', 
    '42': 'Loire', 
    '43': 'Haute-Loire',
    '44': 'Loire-Atlantique', 
    '45': 'Loiret', 
    '46': 'Lot', 
    '47': 'Lot-et-Garonne', 
    '48': 'Lozère',
    '49': 'Maine-et-Loire', 
    '50': 'Manche', 
    '51': 'Marne', 
    '52': 'Haute-Marne', 
    '53': 'Mayenne',
    '54': 'Meurthe-et-Moselle', 
    '55': 'Meuse', 
    '56': 'Morbihan', 
    '57': 'Moselle', 
    '58': 'Nièvre', 
    '59': 'Nord',
    '60': 'Oise', 
    '61': 'Orne', 
    '62': 'Pas-de-Calais', 
    '63': 'Puy-de-Dôme', 
    '64': 'Pyrénées-Atlantiques',
    '65': 'Hautes-Pyrénées', 
    '66': 'Pyrénées-Orientales', 
    '67': 'Bas-Rhin', 
    '68': 'Haut-Rhin', 
    '69': 'Rhône',
    '70': 'Haute-Saône', 
    '71': 'Saône-et-Loire', 
    '72': 'Sarthe', 
    '73': 'Savoie', 
    '74': 'Haute-Savoie',
    '75': 'Paris', 
    '76': 'Seine-Maritime', 
    '77': 'Seine-et-Marne', 
    '78': 'Yvelines', 
    '79': 'Deux-Sèvres',
    '80': 'Somme', 
    '81': 'Tarn', 
    '82': 'Tarn-et-Garonne', 
    '83': 'Var', 
    '84': 'Vaucluse', 
    '85': 'Vendée',
    '86': 'Vienne', 
    '87': 'Haute-Vienne', 
    '88': 'Vosges', 
    '89': 'Yonne', 
    '90': 'Territoire de Belfort',
    '91': 'Essonne', 
    '92': 'Hauts-de-Seine', 
    '93': 'Seine-Saint-Denis', 
    '94': 'Val-de-Marne', 
    '95': 'Val-d\'Oise',
    '971': 'Guadeloupe', 
    '972': 'Martinique', 
    '973': 'Guyane', 
    '974': 'La Réunion', 
    '976': 'Mayotte',
}

## Ci-dessous : on ne propose pas à l'utilisateur les codes de departements, mais plutôt le nom du departement, que j'ai convertie grace au dictionnaire ci-dessus
## Selectbox Departement

sorted_departement = sorted(df['code_departement'].unique())
final_departement = []
final_departement = sorted_departement[:]
final_departement.insert(0, 'Select all')
final_departement2 = ['Select all']
for i in range (1, len(final_departement)):
    departements2 = str(final_departement[i])
    final_departement2.append(dictDepartements[departements2])

departement_choice = st.sidebar.selectbox('Departement', final_departement2)

###### Création de toutes les options de la sidebar

sorted_pieces = sorted(df['nombre_pieces_principales'].dropna().unique())
pieces_choice = st.sidebar.multiselect('Nombre de pièces', sorted_pieces, sorted_pieces)

sorted_type_local = sorted(df["type_local"].dropna().unique())
sorted_type_local = st.sidebar.multiselect('Type de propriété', sorted_type_local, sorted_type_local)

minPrice = int(df['valeur_fonciere'].quantile(q = 0.15)) #ici j'ai preferé éliminer les premiers et derniers 15% des valeurs 
                                                         #du dataset car certaines sont trop extrêmes et faussent la vision globale des données
maxPrice = int(df['valeur_fonciere'].quantile(q = 0.85))
price_choice = st.sidebar.slider('Pannel de prix (en euros)', min_value=minPrice, max_value=maxPrice, value=(minPrice, maxPrice),step=10000)

minSurface = int(df['surface_terrain'].quantile(q = 0.15))
maxSurface = int(df['surface_terrain'].quantile(q = 0.85))
surface_choice = st.sidebar.slider('Surface du bien (en m²)', min_value=minSurface, max_value=maxSurface, value=(minSurface,maxSurface),step=50)

#-----------------------------------#
# Affichage du corps de l'application après sélection des paramètres sur la slidebar par l'utilisateur.
#-----------------------------------#

'''
## Partie 1 : Affichage des données brutes
'''

# Indication à l'utilisateur que le NOM du département (=departement_choice) 
# qu'il a selectionné dans la sidebar correspond à un certain CODE département (key)

if departement_choice!= "Select all":
    st.write("Vous avez choisi le département suivant : " + str(departement_choice))

for key,value in dictDepartements.items():
    if value == departement_choice:
        st.write("Le code département associé dans la base de données est : " + key)
        departement_choice1 = key

## Possibilité de voir les données resultantes de la selection de l'utilisateur 

expander_bar2 = st.beta_expander("Cliquez pour voir les données sélectionnées")

if departement_choice in dictDepartements.values(): #Cas où l'utilisateur a choisi via la sidebar un département spécifique
    myColumns = ['id_mutation','date_mutation', 'code_departement', 'type_local', 'nombre_pieces_principales', 'valeur_fonciere', 'surface_terrain', 'latitude', 'longitude']
    filteredData1 = df[myColumns]
    filteredData1 = filteredData1.loc[filteredData1['code_departement']== int(departement_choice1)]
    filteredData1 = filteredData1.loc[filteredData1['type_local'].isin(sorted_type_local)]
    filteredData1 = filteredData1.loc[filteredData1['nombre_pieces_principales'].isin(pieces_choice)]
    filteredData1 = filteredData1.loc[(filteredData1['valeur_fonciere'] >= min(price_choice)) & (df['valeur_fonciere'] <= max(price_choice))]
    filteredData1 = filteredData1.loc[(filteredData1['surface_terrain'] >= min(surface_choice)) & (df['surface_terrain'] <= max(surface_choice))]
    filteredData1 = filteredData1.dropna()
    expander_bar2.write(filteredData1)
## KPI qui serviront pour la Partie 2 
    taille_dataset=len(filteredData1)
    valeur_fonciere_moyenne = round(filteredData1['valeur_fonciere'].mean())
    surface_moyenne = round(filteredData1['surface_terrain'].mean())
    piece_moyenne= round(filteredData1['nombre_pieces_principales'].mean())

else : #Cas où l'utilisateur a choisi "Select all" via la sidebar 'Département'
    myColumns = ['id_mutation', 'date_mutation', 'code_departement', 'type_local', 'nombre_pieces_principales', 'valeur_fonciere', 'surface_terrain', 'latitude', 'longitude']
    filteredData2 = df[myColumns].dropna()
    filteredData2 = filteredData2.loc[filteredData2['type_local'].isin(sorted_type_local)]
    filteredData2 = filteredData2.loc[filteredData2['nombre_pieces_principales'].isin(pieces_choice)]
    filteredData2 = filteredData2.loc[(filteredData2['valeur_fonciere'] >= min(price_choice)) & (filteredData2['valeur_fonciere'] <= max(price_choice))]
    filteredData2 = filteredData2.loc[(filteredData2['surface_terrain'] >= min(surface_choice)) & (filteredData2['surface_terrain'] <= max(surface_choice))]
    expander_bar2.write(filteredData2)
## KPI qui serviront pour la Partie 2 
    taille_dataset = len(filteredData2)
    valeur_fonciere_moyenne = round(filteredData2['valeur_fonciere'].mean())
    surface_moyenne = round(filteredData2['surface_terrain'].mean())
    piece_moyenne= round(filteredData2['nombre_pieces_principales'].mean())

'''
## Partie 2 : KPI autour des données sélectionnées
'''
# Calculate KPIs (fait au dessus)
if departement_choice in dictDepartements.values():
    listings_count = len(filteredData1["surface_terrain"])
else:
    listings_count = len(filteredData2["surface_terrain"])


# 4 KPI clés concernant les données résultantes de la sélection de l'utilisateur 
st.subheader('Quelques chiffres clés')
metric_row(
    {
        "Transactions" : taille_dataset,
        "Valeur foncière moyenne" : valeur_fonciere_moyenne,
        "Surface moyenne" : surface_moyenne,
        "Nombre de pièce moyen" : piece_moyenne
    }
)

'''
## Partie 3 : Analyse du marché au cours du temps
'''

if departement_choice in dictDepartements.values():
    # Selection unique des dates du dataset resultant de la selection de l'utilisateur  
    d = filteredData1['date_mutation']
    d1 = pd.DatetimeIndex(d).month.astype(str) + "-" + pd.DatetimeIndex(d).year.astype(str)
    d2= sorted(d1.unique())

    nombre_anonces=taille_dataset
    df_inventory = filteredData1[['date_mutation', 'id_mutation']].groupby(['date_mutation']).agg(['nunique']).reset_index()
    df_inventory.columns = ['date_mutation', 'nombre_annonces']

    st.write("Nombre d'annonces au cours du temps")
    st.line_chart(df_inventory.set_index('date_mutation'))

else:
    # Selection unique des dates du dataset resultant de la selection de l'utilisateur
    d = filteredData2['date_mutation']
    d1 = pd.DatetimeIndex(d).month.astype(str) + "-" + pd.DatetimeIndex(d).year.astype(str)
    d2= sorted(d1.unique())

    nombre_anonces=taille_dataset
    df_inventory = filteredData2[['date_mutation', 'id_mutation']].groupby(['date_mutation']).agg(['nunique']).reset_index()
    df_inventory.columns = ['date_mutation', 'nombre_annonces']

    st.write("Nombre d'annonces au cours du temps")
    st.line_chart(df_inventory.set_index('date_mutation'))

'''
## Partie 4 : Zoom sur les données séléctionnées

'''

## Création de datasets qui résument certaines stats du dataset

if departement_choice in dictDepartements.values():
    df_prop_pieces = filteredData1[['id_mutation', 'type_local', 'nombre_pieces_principales']].groupby(['type_local', 'nombre_pieces_principales']).agg(['nunique']).reset_index()
    df_prop_pieces.columns = ['type_local', 'nombre_pieces_principales', 'nombre_annonces']
    df_prop_pieces['nombre_pieces_principales'] = df_prop_pieces['nombre_pieces_principales'].astype(str) + ' Pièce(s) principale(s)'
else:
    df_prop_pieces = filteredData2[['id_mutation', 'type_local', 'nombre_pieces_principales']].groupby(['type_local', 'nombre_pieces_principales']).agg(['nunique']).reset_index()
    df_prop_pieces.columns = ['type_local', 'nombre_pieces_principales', 'nombre_annonces']
    df_prop_pieces['nombre_pieces_principales'] = df_prop_pieces['nombre_pieces_principales'].astype(str) + ' Pièce(s) principale(s)'

## Création d'un dataset individuel pour chaque type de local/proprieté
df_apt_pieces = df_prop_pieces[df_prop_pieces['type_local'] == 'Appartement']
df_house_pieces = df_prop_pieces[df_prop_pieces['type_local'] == 'Maison']
df_dep_pieces = df_prop_pieces[df_prop_pieces['type_local'] == 'Dépendance']
df_local_pieces = df_prop_pieces[df_prop_pieces['type_local'] == 'Local industriel. commercial ou assimilé']

def show_nonempty_df(df):
    if not df.empty:
        st.write(df)

## Datasets qui résument les informations du dataset selon le type de local

def dataResume():
        agreeResume = st.checkbox('Voir un resumé des données par type de bien')
        if agreeResume:
            show_nonempty_df(df_apt_pieces)
            show_nonempty_df(df_house_pieces)
            show_nonempty_df(df_dep_pieces)
            show_nonempty_df(df_local_pieces)
dataResume()

###### Visualisation d'une carte de la France marquée par la position des biens du dataset

if departement_choice in dictDepartements.values():
    def dataCarte():
        agreeCarte = st.checkbox('Voir la carte')
        if agreeCarte:
            map_data = filteredData1[['latitude', 'longitude']]
            st.header("Visualisation géographiques des biens sélectionnés")
            st.map(map_data)
    dataCarte()
else :
    def dataCarte():
        agreeCarte = st.checkbox('Voir la carte')
        if agreeCarte:
            map_data = filteredData2[['latitude', 'longitude']]
            st.header("Visualisation géographiques des biens sélectionnés")
            st.map(map_data)
    dataCarte()

### Histogramme confrontant latitude et longitude des données selectionnées

if departement_choice in dictDepartements.values():
    def dataHistLonLat():
        agreeHistLonLat = st.checkbox('Voir la répartition des Longitudes et Latitudes')
        if agreeHistLonLat:
            plt.hist(filteredData1['longitude'], bins=5, range=(min(filteredData1['longitude']),max(filteredData1['longitude'])), color= 'g', alpha = 0.5, label = 'Longitude')
            plt.legend(loc = 'best')
            plt.twiny()
            plt.hist(filteredData1['latitude'], bins=5, range=(min(filteredData1['latitude']),max(filteredData1['latitude'])), color = 'r', alpha = 0.5, label = 'Latitude')
            plt.legend(loc = 'upper left')
            plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
    dataHistLonLat()
else:
    def dataHistLonLat():
        agreeHistLonLat = st.checkbox('Voir la répartition des Longitudes et Latitudes')
        if agreeHistLonLat:
            plt.hist(filteredData2['longitude'], bins=100, range=(min(filteredData2['longitude']),max(filteredData2['longitude'])), color= 'g', alpha = 0.5, label = 'Longitude')
            plt.legend(loc = 'best')
            plt.twiny()
            plt.hist(filteredData2['latitude'], bins=100, range=(min(filteredData2['latitude']),max(filteredData2['latitude'])), color = 'r', alpha = 0.5, label = 'Latitude')
            plt.legend(loc = 'upper left')
            plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
    dataHistLonLat()

'''
## Partie 5 : Prédiction à l'aide du Machine Learning

'''

### L'algorithme a été trouvé sur un site internet, je l'ai adapté à notre Dataset. 
### J'ai trouvé ça pertinent de l'ajouter ici à la fin de la sidebar, cela rajoute une petite option en plus 

st.sidebar.subheader('Prédiction du prix de la maison de vos rêves')
# Options de la sidebar :
params={
'nombre_pieces_principales' : st.sidebar.selectbox('Nombre de pièces',(0.0,1.0,2.0,3.0,4.0,5.0,6.0,8.0)),
'surface_terrain' : st.sidebar.slider('Surface du terrain (en m²)',minSurface,maxSurface,step=100),
}

## imports 

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from math import sqrt

## models

def map_df(df):
	df=df[df['nombre_pieces_principales']==params['nombre_pieces_principales']]
	df=df[df['surface_terrain']==params['surface_terrain']]
	df.reset_index()
	return df


test_size=st.sidebar.slider('Choisissez le Test Size', 0.05,0.5,0.25,step=0.05)


@st.cache
def get_models():
    dfff=df[['valeur_fonciere','nombre_pieces_principales','surface_terrain']].dropna()
    y=dfff['valeur_fonciere']
    X=dfff[['nombre_pieces_principales','surface_terrain']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
    models = [DummyRegressor(strategy='mean'),
			   RandomForestRegressor(n_estimators=170,max_depth=25),
			   DecisionTreeRegressor(max_depth=25),
			   GradientBoostingRegressor(learning_rate=0.01,n_estimators=200,max_depth=5), 
			   LinearRegression(n_jobs=10, normalize=True)]
    df_models = pd.DataFrame()
    temp = {}
    print(X_test)

    for model in models:
	    print(model)
	    m = str(model)
	    temp['Model'] = m[:m.index('(')]
	    model.fit(X_train, y_train)
	    temp['RMSE_Price'] = sqrt(mse(y_test, model.predict(X_test)))
	    temp['Pred Value']=model.predict(pd.DataFrame(params,  index=[0]))[0]
	    print('RMSE score',temp['RMSE_Price'])
	    df_models = df_models.append([temp])
    df_models.set_index('Model', inplace=True)
    pred_value=df_models['Pred Value'].iloc[[df_models['RMSE_Price'].argmin()]].values.astype(float)
    return pred_value, df_models

## Bar de progression : 

def run_status():
	latest_iteration = st.empty()
	bar = st.progress(0)
	for i in range(100):
		latest_iteration.text(f'Percent Complete {i+1}')
		bar.progress(i + 1)
		time.sleep(0.1)
		st.empty()
run_status()

## Resultat de la prédiction

def run_data():
	df_models=get_models()[0][0]
	st.write("D'après les paramètres que vous avez indiqué, la valeur prédite du bien est de **${:.2f}**".format(df_models))

## Bouton Prédiction : lance la prédiction après saisi des parametres par l'utilisateur 

btn = st.sidebar.button("Prédire")
if btn:
	run_data()
else:
	pass