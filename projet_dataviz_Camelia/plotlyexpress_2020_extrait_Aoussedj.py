#page à part (s'ouvre en pop-up)
import plotly.express as px
import pandas as pd

dff = pd.read_csv("full_2020_extrait.csv")
df=dff[['valeur_fonciere','surface_terrain','code_departement','date_mutation']].dropna()


fig = px.scatter(df, x="surface_terrain", y="valeur_fonciere", animation_frame="date_mutation", animation_group="code_departement",
           size="valeur_fonciere", color="valeur_fonciere", hover_name="code_departement",
           log_x=True, size_max=55, range_x=[1e+01,91893], range_y=[1e+01,1e+06])

fig["layout"].pop("updatemenus")
fig.show()