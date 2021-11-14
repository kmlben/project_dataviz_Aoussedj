#page à part (s'ouvre en pop-up)

import plotly.express as px
import pandas as pd

dff = pd.read_csv("full_2020.csv")
df=dff[['valeur_fonciere','surface_terrain','code_departement','date_mutation']].dropna()


fig = px.scatter(df, x="surface_terrain", y="valeur_fonciere", animation_frame="date_mutation", animation_group="code_departement",
           size="valeur_fonciere", color="valeur_fonciere", hover_name="code_departement",
           log_x=True, size_max=55, range_x=[df['surface_terrain'].quantile(q = 0.25),df['surface_terrain'].quantile(q = 0.75)], range_y=[df['valeur_fonciere'].quantile(q = 0.25),df['valeur_fonciere'].quantile(q = 0.25)])
# elimination du premier et dernier quartile car valeurs trop marginales, donc visualisation des données pas optimale

fig["layout"].pop("updatemenus")
fig.show()