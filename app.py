from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import plotly.express as px
import plotly
import json
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go

app = Flask(__name__)

df=pd.read_csv('world_population.csv')
df.sort_values(by=['2022 Population'],ascending=False,inplace=True)

df1=pd.read_csv('population_total_long.csv')

total_2022 = df['2022 Population'].sum()
total_2020 = df['2020 Population'].sum()
total_2015 = df['2015 Population'].sum()
total_2010 = df['2010 Population'].sum()
total_2000 = df['2000 Population'].sum()
total_1990 = df['1990 Population'].sum()
total_1980 = df['1980 Population'].sum()
total_1970 = df['1970 Population'].sum()

world_pop_all = [total_2022, total_2020, total_2015,total_2010,total_2000,total_1990,total_1980,total_1970]
world_pop_all = world_pop_all[::-1]
model=LinearRegression()
y=np.asarray(world_pop_all).reshape(-1,1)
x=np.asarray([int(i) for i in ['1970','1980','1990','2000','2010','2015','2020','2022']]).reshape(-1,1)
model.fit(x,y)
years=[]
pop=[]
for i in range(1970,2051,5):
  years.append([i])

pop=model.predict(years)
years=[i[0] for i in years]
pop=[i[0] for i in pop]


df_i = pd.read_csv('indian population new.csv')

df_1 = pd.read_csv('new.csv')
world_urban = px.choropleth(df_1.dropna(), locations="CCA3", color="Urb Pop (%)", title="urban population %",
                    color_continuous_scale=[(0, 'black'), (0.001, 'white'), (0.0010, 'red'), (0.1, 'yellow'), (1, 'green')],template="plotly_dark")
world_urban.update_layout(height=700, width=1000)
world_urban_json= json.dumps(world_urban, cls=plotly.utils.PlotlyJSONEncoder)

#Shreyansh
list1 = ['Africa','Asia','Europe','North America','Oceania','South America']
Continents = np.array(list1)
total_pop = df.groupby('Continent').sum()
total_pop.drop(total_pop.iloc[:,9:14],inplace=True,axis=1)
total_pop.drop(['Rank'],inplace=True,axis=1)
total_africa_pop = total_pop.iloc[0,:].sum()
total_asia_pop = total_pop.iloc[1,:].sum()
total_europe_pop = total_pop.iloc[2,:].sum()
total_north_america_pop = total_pop.iloc[3,:].sum()
total_oceania_pop = total_pop.iloc[4,:].sum()
total_south_america_pop = total_pop.iloc[5,:].sum()
Overall_continent_pop = [total_africa_pop,total_asia_pop,total_europe_pop,total_north_america_pop,total_oceania_pop,total_south_america_pop]
df_cont = {'All_continents':Continents,'Total_population_All_Time':Overall_continent_pop}
df_cont = pd.DataFrame(data=df_cont)
cont_total_pop = px.bar(df_cont,x="All_continents",y="Total_population_All_Time",template="plotly_dark",title="Population of continents for all time")
cont_total_pop.update_layout(height=400, width=600, margin=dict(l=0, r=0, t=50, b=0))
cont_total_pop_json= json.dumps(cont_total_pop, cls=plotly.utils.PlotlyJSONEncoder)

df2 = df.groupby('Continent').sum()
total_growth_africa = df2.iloc[0,11]
total_growth_asia = df2.iloc[1,11]
total_growth_europe = df2.iloc[2,11]
total_growth_n_america = df2.iloc[3,11]
total_growth_oceania = df2.iloc[4,11]
total_growth_s_america = df2.iloc[5,11]

Overall_growth_rate = [total_growth_africa,total_growth_asia,total_growth_europe,total_growth_n_america,total_growth_oceania,total_growth_s_america ]

df3 = {'All_continents':Continents,'Overall_growth_rate':Overall_growth_rate}
df3 = pd.DataFrame(data=df3)
cont_growth = px.histogram(df3,x="All_continents",y="Overall_growth_rate",color="All_continents",title="Population Growth Rate of Continents",template="plotly_dark")
cont_growth.update_layout(height=400, width=600, margin=dict(l=0, r=0, t=50, b=0))
cont_growth_json= json.dumps(cont_growth, cls=plotly.utils.PlotlyJSONEncoder)

def find_country_pop_from_growth(n):
    condition = df['Country/Territory'] == n
    cpop_2020 = df.loc[condition, '2020 Population'].values[0]
    cpop_2022 = df.loc[condition, '2022 Population'].values[0]
    c_growth_rate = ((cpop_2022 - cpop_2020)/cpop_2020)*100

    print(c_growth_rate)
    print(cpop_2022+(cpop_2022*c_growth_rate/100))
    pop_li=[]
    years_li=[]
    current = cpop_2022
    for i in range(1,29):
        pop = current + (current*c_growth_rate/100)
        pop_li.append(pop)
        years_li.append(2022+i)
        current=pop
    return(pop_li,years_li)


@app.route('/',methods=('GET','POST'))
def home():

    c_name=df.head(15)['Country/Territory'].to_list()
    pop_2022 = df.head(15)['2022 Population'].to_list()
    c_name.append('Remaining')
    pop_2022.append(df.iloc[14:,:]['2022 Population'].sum())
    temp=pd.DataFrame({'Name':c_name,'pop':pop_2022})
    top_15 = px.pie(temp,values='pop', names='Name', title='Population of top 15 countries',template="plotly_dark")
    top_15.update_layout(height=400, width=600, margin=dict(l=0, r=0, t=50, b=0))
    top_15_json= json.dumps(top_15, cls=plotly.utils.PlotlyJSONEncoder)
    
    pop_density = px.choropleth(df, locations="CCA3", color="Density (per km²)", title="Population Density each country",
                    color_continuous_scale=[(0, 'blue'), (0.001, 'green'), (0.005, 'yellow'), (0.1, 'orange'), (1, 'red')],template="plotly_dark")
    pop_density.update_layout(height=700, width=1000)
    pop_density_json= json.dumps(pop_density, cls=plotly.utils.PlotlyJSONEncoder)


    continents_2022=df.groupby('Continent')['2022 Population'].sum().reset_index()
    continent_pop = px.pie(continents_2022, values='2022 Population', names='Continent',title='2022 Population Continent Wise',template="plotly_dark")
    continent_pop.update_layout(height=400, width=600, margin=dict(l=0, r=0, t=50, b=0))
    continent_pop_json= json.dumps(continent_pop, cls=plotly.utils.PlotlyJSONEncoder)
    
    
    india_pop=px.bar(df_i, x='Year',y='Population',title="India's population Growth",orientation='v',color='Year')
    gap_data =df_i[10::10]
    india_urb_rur = go.Figure(data=[
        go.Bar(name='Urban', x=gap_data['Year'], y=gap_data['Urban Population % of Total Population']),
        go.Bar(name='Rural', x=gap_data['Year'], y=gap_data['Rural Population % of Total Population'])
    ])
    # Change the bar mode
    india_urb_rur.update_layout(barmode='group',title='Urban v/s Rural Population over the years')

    india_inf = px.line(df_i,x='Year',y='Infant Mortality Rate',title='Infant Mortality decline in India')

    india_birth_death = px.line(df_i,x='Year',y=['Birth Rate','Death Rate'],title='Birth Rate and Death Rate')
    correlation_matrix = df_i[['Population Density','Life Expectancy','Birth Rate','Death Rate','Infant Mortality Rate','Fertility Rate','Net Migration Rate']].corr()
    india_corr = px.imshow(correlation_matrix,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    color_continuous_scale='Inferno',
                    title='Correlation Matrix of Various Feeatures for India',template="plotly_dark")
    india_corr.update_layout(height=500,width=700,margin=dict(l=0, r=0, t=50, b=0))
    all=[]
    for i in [india_pop,india_urb_rur,india_inf, india_birth_death]:
        i.update_layout(height=400, width=600, margin=dict(l=0, r=0, t=50, b=0),title_x=0.5,template="plotly_dark")
        js= json.dumps(i, cls=plotly.utils.PlotlyJSONEncoder)
        all.append(js)
    
    all.append(json.dumps(india_corr, cls=plotly.utils.PlotlyJSONEncoder))
    

    if request.method=="POST":
        n=request.form['c_name']
        try:
            temp=df1[df1['Country Name'] == n]
        except:
            print("Not Found")
            return redirect(url_for('home'))
        try:
            pop_growth_rate,year_growth_rate = find_country_pop_from_growth(n)
        except:
            return redirect(url_for('home'))
        x=temp['Year'].to_list()
        y=temp['Count'].to_list()

        x=[[i] for i in x]
        y=[[j] for j in y]

        model=LinearRegression()
        model.fit(x,y)

        years_c=[]
        pop_c=[]
        for i in range(1960,2051,5):
            years_c.append([i])

        pop_c=model.predict(years_c)
        years_c=[i[0] for i in years_c]
        pop_c=[i[0] for i in pop_c]
        
        pred2=px.line(x=years_c, y=pop_c, markers=True,title=f"{n}'s Population Prediction",template="plotly_dark")
        pred2.add_scatter(x=temp['Year'].to_list(),y=temp['Count'].to_list(),name='Actual Data')
        pred2.add_scatter(y=pop_growth_rate, x=year_growth_rate,name='Population based on current Growth Rate')

        pred2.update_layout(height=400, width=800, margin=dict(l=0, r=0, t=50, b=0))
        pred_json= json.dumps(pred2, cls=plotly.utils.PlotlyJSONEncoder)

        
        return render_template('index.html', pred_json=pred_json, 
        top_15_json=top_15_json, 
        pop_density_json=pop_density_json, 
        continent_pop_json=continent_pop_json, 
        india_pop_json=all[0], 
        india_urb_json = all[1], 
        india_inf_json=all[2], 
        india_birth_death_json=all[3], 
        india_corr_json = all[4], 
        world_urban_json = world_urban_json, 
        cont_growth_json=cont_growth_json, 
        cont_total_pop_json = cont_total_pop_json
        )

    pred1=px.line(x=years, y=pop, markers=True,title="World's Population Prediction",template="plotly_dark")
    pred1.add_scatter(x=['1970','1980','1990','2000','2010','2015','2020','2022'],y=world_pop_all)
    pred1.update_layout(height=400, width=800, margin=dict(l=0, r=0, t=50, b=0))

    pred_json= json.dumps(pred1, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html', pred_json=pred_json, 
    top_15_json=top_15_json, 
    pop_density_json=pop_density_json, 
    continent_pop_json=continent_pop_json, 
    india_pop_json=all[0], 
    india_urb_json = all[1], 
    india_inf_json=all[2], 
    india_birth_death_json=all[3], 
    india_corr_json = all[4], 
    world_urban_json = world_urban_json, 
    cont_growth_json=cont_growth_json, 
    cont_total_pop_json = cont_total_pop_json
    )


