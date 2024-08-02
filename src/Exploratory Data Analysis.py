#Before diving into the EDA, we import some libraries and also present some helper functions and commands that will be utilized further down the road for visualizations.

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import ListedColormap, LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def CustomCmap(from_rgb,to_rgb):

    # from color r,g,b
    r1,g1,b1 = from_rgb

    # to color r,g,b
    r2,g2,b2 = to_rgb

    cdict = {'red': ((0, r1, r1),
                   (1, r2, r2)),
           'green': ((0, g1, g1),
                    (1, g2, g2)),
           'blue': ((0, b1, b1),
                   (1, b2, b2))}

    cmap = LinearSegmentedColormap('custom_cmap', cdict)
    return cmap

mycmap = CustomCmap([1.0, 1.0, 1.0], [72/255, 99/255, 147/255])
mycmap_r = CustomCmap([72/255, 99/255, 147/255], [1.0, 1.0, 1.0])

mycol = (72/255, 99/255, 147/255)
mycomplcol = (129/255, 143/255, 163/255)
othercol1 = (135/255, 121/255, 215/255)
othercol2 = (57/255, 119/255, 171/255)
othercol3 = (68/255, 81/255, 91/255)
othercol4 = (73/255, 149/255, 139/255)

#Evolution of top countries with respect to mortality
#Herein, the mortality rate is calculated as the total number of deaths divided by each location's population (another common definition is the total number of deaths by Covid divided by the total number of Covid cases). For this purpose, a column named mortality is constructed. Using this column, we identify the top 10 countries in terms of mortality rates, for every day of the studied time interval.

dates_frame = df.select("date").distinct().orderBy('date').collect()
dates_list = [str(dates_frame[x][0]) for x in range(len(dates_frame))]

df_for_mort = df.filter(F.col('population') != 0.0).withColumn("mortality", F.col("total_deaths")/F.col("population"))

for i, this_day in enumerate(dates_list):
    this_day_top_10 = df_for_mort.filter(F.col('date') == this_day).orderBy("mortality", ascending=False).select(["location","mortality"]).take(10)
    if i == 0:
        ct_list = [(this_day_top_10[x][0],this_day_top_10[x][1]) for x in range(10)]
        print("During "+this_day+", the top 10 countries with the highest mortality rate were:")
        for country, instance in ct_list:
            print(f"▶ {country}, with mortality rate {100*instance:.2f}%.")
        new_set = set(ct_list[x][0] for x in range(10))
    elif i == len(dates_list)-1:
        ct_list = [(this_day_top_10[x][0],this_day_top_10[x][1]) for x in range(10)]
        print("During "+this_day+", the top 10 countries with the highest mortality rate were:")
        for country, instance in ct_list:
            print(f"▶ {country}, with mortality rate {100*instance:.2f}%.")
    else:
        new_set = set(this_day_top_10[x][0] for x in range(10))
        if new_set != old_set:
            left_out = old_set-new_set
            new_additions = new_set-old_set
            print("This was the top ten until "+this_day+", when "+", ".join(str(s) for s in new_additions)+" joined the list, replacing "+", ".join(str(s) for s in left_out)+".")
    new_set, old_set = set(), new_set


#Evolution of top countries with respect to total cases per million
#The same procedure can be performed for the number of total cases per million. We choose to normalize the total number of cases in this way in order to be able to compare locations with different populations.

for i, this_day in enumerate(dates_list):
    this_day_top_10 = df.filter(F.col('date') == this_day).orderBy("total_cases_per_million", ascending=False).select(["location","total_cases_per_million"]).take(10)
    if i == 0:
        ct_list = [(this_day_top_10[x][0],this_day_top_10[x][1]) for x in range(10)]
        print("During "+this_day+", the top 10 countries with the highest number of total cases per million were:")
        for country, instance in ct_list:
            print(f"▶ {country}, with {instance} total cases per million.")
        new_set = set(ct_list[x][0] for x in range(10))
    elif i == len(dates_list)-1:
        ct_list = [(this_day_top_10[x][0],this_day_top_10[x][1]) for x in range(10)]
        print("During "+this_day+", the top 10 countries with the highest number of total cases per million were:")
        for country, instance in ct_list:
            print(f"▶ {country}, with {instance} total cases per million.")
    else:
        new_set = set(this_day_top_10[x][0] for x in range(10))
        if new_set != old_set:
            left_out = old_set-new_set
            new_additions = new_set-old_set
            print("This was the top ten until "+this_day+", when "+", ".join(str(s) for s in new_additions)+" joined the list, replacing "+", ".join(str(s) for s in left_out)+".")
    new_set, old_set = set(), new_set

#Hospitalized Patients and ICU Admissions
#Moving on, we study the hosp_patients and icu_patients features by visualizing the corresponding timeseries for the total number of hospitalized and ICU patients on a global scale.

dt_ord = df.orderBy("date", ascending=True).groupBy("date")

hosps = dt_ord.agg(F.sum("hosp_patients")).collect()
hosps = [hosps[i][1] for i in range(len(hosps))]

icus = dt_ord.agg(F.sum("icu_patients")).collect()
icus = [icus[i][1] for i in range(len(icus))]

sns.set(style = "darkgrid")

alt_dts_list = [dt.replace('2021-', '') for dt in dates_list]
tick_marks = np.arange(len(alt_dts_list))

fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(14,5))

for pat, col, style, ax, where in zip([hosps,icus], [mycol, mycomplcol],
                                      ['solid', 'dashed'], [ax1,ax2], ['Normal Beds','ICUs']): 
    ax.plot(alt_dts_list, pat, linestyle=style, color=col)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Patients")
    ax.set_title(f"Daily Number of Patients in {where}", fontsize=14)
    ax.set_xticks(tick_marks[::5])
    ax.set_xticklabels(alt_dts_list[::5], rotation=45)
    
plt.show()

matplotlib.rc_file_defaults()

#Geographic Heatmap of Total Cases
#An interesting visualization is the geographic heatmap, which is a 2D representation of countries world-wide which are colored depending on their intensity as far as a specific feature is concerned. Below, we construct the geographic heatmap for the number of total cases on a global scale. A heatmap image is extracted for each day and afterwards all images are merged into a .gif file. The heatmap is constructed using the geopandas library, as seen below. Note that to do this, we must first download a shapefile (.shp) which is the foundation for the construction of the heatmap and can be found here.

import requests, zipfile
from io import BytesIO

zip_file_url = "https://srigas.me/kaggle/owid-nb-data.zip"

request = requests.get(zip_file_url)
zipDocument = zipfile.ZipFile(BytesIO(request.content))

zipDocument.extractall()

import geopandas as gpd

shapefile = 'countries.shp'
geo_df = gpd.read_file(shapefile)[['ADMIN','ADM0_A3','geometry']]
geo_df.columns = ['location', 'iso_code', 'geometry']
geo_df = geo_df.drop(geo_df.loc[geo_df['location'] == 'Antarctica'].index) # exclude Antarctica

print('Initializing the construction of heatmaps for every day.')

ct = 0
for this_day in dates_list:
    # The conversion of the required columns into a Pandas df is necessary to perform the mapping
    day_df = df.filter(F.col('date') == this_day).select(["iso_code","total_cases"]).toPandas()

    merged_df = pd.merge(left=geo_df, right=day_df, how='left', left_on='iso_code', right_on='iso_code')

    title = f'Total COVID-19 Cases as of {this_day}'
    col = 'total_cases'
    vmin, vmax = merged_df[col].min(), merged_df[col].max()
    cmap = mycmap
    divnorm = TwoSlopeNorm(vcenter=0.08*20365726)

    # Create figure and axes for Matplotlib
    fig, ax = plt.subplots(1, figsize=(20, 8))

    # Remove the axis
    ax.axis('off')
    merged_df.plot(column=col, ax=ax, edgecolor='1.0', linewidth=1, norm=divnorm, cmap=cmap)

    # Add a title
    ax.set_title(title, fontdict={'fontsize': '25', 'fontweight': '3'})

    # Create colorbar as a legend
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)

    # Empty array for the data range
    sm._A = []

    # Add the colorbar to the figure
    cbaxes = fig.add_axes([0.15, 0.25, 0.01, 0.4])
    cbar = fig.colorbar(sm, cax=cbaxes)
    plt.savefig(f'world_map_{this_day}.png', bbox_inches='tight')
    plt.close(fig)
    ct += 1

print(f'Process complete. {ct} heatmap(s) were extracted, ready to be converted into a .gif file.')

#Geographic Correlation of Excess Mortality
#Based on the previous visualization it appears that some neighbouring countries are correlated with respect to the total number of cases (for example France and Germany). A reasonable hypothesis is that the same may be true for other features as well, such as the excess mortality.

#The excess mortality is a feature for which the reports are weekly and not daily. It is equal to the total number of deaths for a specific week minus the mean number of deaths, based on reports from previous years. While it is not a feature directly connected with Covid, it's expected that during a global pandemic the excess mortality can be mainly attributed to this pandemic.

#In order to investigate the correlation between neighbouring countries, we must first develop a list of dates for which reports on excess mortality are available (for all other dates, the entries are equal to zero due to our preprocessing).

exc_dates_list = df.filter(F.col('excess_mortality') != 0.0).select(['date']).distinct().orderBy('date').collect()
exc_dates_list = [str(exc_dates_list[i][0]) for i in range(len(exc_dates_list))]

print('Initializing the construction of heatmaps for every day.')

ct = 0
for this_day in exc_dates_list:
    europe_df = df.filter(F.col('date') == this_day).filter(F.col('continent') == 'Europe').filter(F.col('excess_mortality') != 0.0).select(["iso_code","excess_mortality"])
    
    geo_eu = pd.merge(left=geo_df, right=europe_df.toPandas(), how='inner', on='iso_code')

    fig, ax = plt.subplots(1,1)

    col = 'excess_mortality'
    cmap = mycmap

    vmin, vmax = geo_eu[col].min(), geo_eu[col].max()
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)

    ax.axis('off')
    ax.axis([-13, 44, 33, 72])
    geo_eu.plot(column=col, ax=ax, edgecolor='1.0', linewidth=1, norm=None, cmap=cmap)
    ax.set_title(f'Excess Mortality in Europe as of {this_day}', fontdict={'fontsize': '14', 'fontweight': '3'})
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=.2)
    fig.add_axes(cax)
    fig.colorbar(sm, cax=cax)
    plt.savefig(f'europe_{this_day}.png', bbox_inches='tight')
    plt.close(fig)
    ct += 1
    
print(f'Process complete. {ct} heatmap(s) were extracted, ready to be converted into a .gif file.')

european_df = df.filter(F.col('continent') == 'Europe').filter(F.col('excess_mortality') != 0.0)
european_cts = european_df.select(['location']).distinct().collect()
european_cts = [european_cts[i][0] for i in range(len(european_cts)) if european_df.filter(F.col('location') == european_cts[i][0]).count() == len(exc_dates_list)]
print(f'{len(european_cts)} European countries are chosen for this analysis.')

from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark.sql.window import Window

eu_cts_df = european_df.filter(F.col('location') == european_cts[0]).select(['excess_mortality']).withColumnRenamed("excess_mortality", european_cts[0])
# required for the proper join of the following DataFrames
eu_cts_df = eu_cts_df.withColumn('row_index', row_number().over(Window.partitionBy(F.lit(0)).orderBy(monotonically_increasing_id())))

for country in european_cts[1:]:
    new_ct_df = european_df.filter(F.col('location') == country).select(['excess_mortality']).withColumnRenamed("excess_mortality", country)
    new_ct_df = new_ct_df.withColumn('row_index', row_number().over(Window.partitionBy(F.lit(0)).orderBy(monotonically_increasing_id())))
    
    eu_cts_df = eu_cts_df.join(new_ct_df, on=["row_index"])
    
eu_cts_df = eu_cts_df.drop("row_index")


from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

vector_col = "corr_features"
assembler = VectorAssembler(inputCols=eu_cts_df.columns, outputCol=vector_col)
df_vector = assembler.transform(eu_cts_df).select(vector_col)

matrix = Correlation.corr(df_vector, vector_col, 'pearson')
cor_np = matrix.collect()[0][matrix.columns[0]].toArray()

fig, ax = plt.subplots(figsize=(13,10))

sns.heatmap(cor_np, linewidths=.5, ax=ax, vmin=0.8, vmax=1, cmap=mycmap,
            xticklabels=european_cts, yticklabels=european_cts)
ax.set_title('Correlation Matrix for Excess Mortality values per Country', fontsize=16)
plt.show()


#Reproduction Rate on the Continent Level
#Moving on, we continue the study of the pandemic's features on a geographic viewpoint by grouping the countries together into continents. For this purpose, the DataFrame is split into continent-level DataFrames, in order to be able to draw the geographic heatmaps separately. The studied feature is now the daily reproduction rate, corresponding to the heatmap's intensity, with a common scale in order to be able to compare different continents. As usual, a .gif image is constructed using separate heatmap images for each day in the January-February interval.

#Note that all filterings are performed using PySpark, however the joining of the DataFrames is performed using Pandas because PySpark cannot recognize the column that corresponds to each location's geometry, which is required for the construction of the geographic maps.

daily_means = {'AS': [], 'EU' : [], 'NAM' : [], 'SAM' : [], 'OC' : [], 'AF' : []}

print('Initializing the construction of heatmaps for every day.')

ct = 0
for this_day in dates_list:
    asia_df = df.filter(F.col('date') == this_day).filter(F.col('continent') == 'Asia').filter(F.col('reproduction_rate') != 0.0).select(["iso_code","reproduction_rate"])
    europe_df = df.filter(F.col('date') == this_day).filter(F.col('continent') == 'Europe').filter(F.col('reproduction_rate') != 0.0).select(["iso_code","reproduction_rate"])
    namerica_df = df.filter(F.col('date') == this_day).filter((F.col('continent') == 'North America')).filter(F.col('reproduction_rate') != 0.0).select(["iso_code","reproduction_rate"])
    samerica_df = df.filter(F.col('date') == this_day).filter((F.col('continent') == 'South America')).filter(F.col('reproduction_rate') != 0.0).select(["iso_code","reproduction_rate"])
    oceania_df = df.filter(F.col('date') == this_day).filter(F.col('continent') == 'Oceania').filter(F.col('reproduction_rate') != 0.0).select(["iso_code","reproduction_rate"])
    africa_df = df.filter(F.col('date') == this_day).filter(F.col('continent') == 'Africa').filter(F.col('reproduction_rate') != 0.0).select(["iso_code","reproduction_rate"])

    daily_means['AS'].append(asia_df.select(F.mean(F.col('reproduction_rate'))).collect()[0][0])
    daily_means['EU'].append(europe_df.select(F.mean(F.col('reproduction_rate'))).collect()[0][0])
    daily_means['NAM'].append(namerica_df.select(F.mean(F.col('reproduction_rate'))).collect()[0][0])
    daily_means['SAM'].append(samerica_df.select(F.mean(F.col('reproduction_rate'))).collect()[0][0])
    daily_means['OC'].append(oceania_df.select(F.mean(F.col('reproduction_rate'))).collect()[0][0])
    daily_means['AF'].append(africa_df.select(F.mean(F.col('reproduction_rate'))).collect()[0][0])

    geo_as = pd.merge(left=geo_df, right=asia_df.toPandas(), how='inner', on='iso_code')
    geo_eu = pd.merge(left=geo_df, right=europe_df.toPandas(), how='inner', on='iso_code')
    geo_sam = pd.merge(left=geo_df, right=samerica_df.toPandas(), how='inner', on='iso_code')
    geo_nam = pd.merge(left=geo_df, right=namerica_df.toPandas(), how='inner', on='iso_code')
    geo_oc = pd.merge(left=geo_df, right=oceania_df.toPandas(), how='inner', on='iso_code')
    geo_af = pd.merge(left=geo_df, right=africa_df.toPandas(), how='inner', on='iso_code')

    fig, axes = plt.subplots(2,3, figsize=(18,14))

    col = 'reproduction_rate'
    cmap = mycmap

    vmin = min(geo_as[col].min(),geo_eu[col].min(),geo_sam[col].min(),geo_nam[col].min(),geo_oc[col].min(),geo_af[col].min())
    vmax = max(geo_as[col].max(),geo_eu[col].max(),geo_sam[col].max(),geo_nam[col].max(),geo_oc[col].max(),geo_af[col].max())
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)

    for ax, data, cont, dims in zip(axes.flat,
                               [geo_eu, geo_nam, geo_af, geo_as, geo_sam, geo_oc],
                               ['Europe','North America','Africa','Asia','South America','Oceania'],
                               [[-13, 44, 33, 72],[-170, -50, 5, 85],[-20, 55, -38, 40],[25, 145, -10, 60],[-85, -32, -58, 15],[110,160,-45,0]]):
        title = f'{cont}'
        ax.axis('off')
        ax.axis(dims)
        data.plot(column=col, ax=ax, edgecolor='1.0', linewidth=1, norm=None, cmap=cmap)
        ax.set_title(title, fontdict={'fontsize': '18', 'fontweight': '3'})

    clb = fig.colorbar(sm, ax=axes.flat, location='bottom', fraction=0.056)
    clb.ax.set_title(f'COVID-19 Reproduction Rate as of {this_day}', fontsize=22)
    plt.savefig(f'cont_maps_{this_day}.png', bbox_inches='tight')
    plt.close(fig)
    ct += 1

print(f'Process complete. {ct} heatmap(s) were extracted, ready to be converted into a .gif file.')


sns.set(style = "darkgrid")

fig, ax = plt.subplots(1, 1, figsize=(10,5))

for key, col, lab in zip(daily_means,
                             [othercol1, othercol2, othercol3, mycol, mycomplcol, othercol4],
                             ['Asia', 'Europe', 'Africa', 'N. America', 'S. America', 'Oceania']): 
    ax.plot(alt_dts_list, daily_means[key], color=col, label = lab)
    
ax.set_xlabel("Date")
ax.set_ylabel("Reproduction Rate")
ax.set_title("Daily Mean Reproduction Rate per Continent", fontsize=14)
tick_marks = np.arange(len(alt_dts_list))
ax.set_xticks(tick_marks[::5])
ax.set_xticklabels(alt_dts_list[::5], rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, title="Continent")
plt.show()

matplotlib.rc_file_defaults()


#Correlation between different features
#Moving away from the geographic visualizations we proceed to study the correlation between different features on the country level. More specifically, we study the correlation between excess mortality and number of daily tests performed. The columns used for this purpose are excess_mortality and new_tests_smoothed. The reason why new_tests_smoothed is used instead of new_tests is because it contains less missing values compared to new_tests. Since the analysis is performed on the country level, there is no reason to use normalized features, since the final metric is simply the correlation between different features corresponding to the same country.

#It's important to note that not all countries are taken into account, but only these with more than 5 non zero entries for the excess_mortality feature, for the reasons discussed previously as far as this feature is concerned. Without this filter, a lot of countries seem to show the maximum/minimum value of +1/-1 for the studied correlation, simply because there are very few entries. A typical example is that of Albania, for which the correlation is found equal to +1, simply because it has only 2 entries for the excess_mortality feature during the months January-February 2021.

countries_frame = df.select("location").distinct().collect()
exclusion_list = ['Europe', 'World', 'Asia', 'North America', 'South America', 'Africa', 'Oceania', 'Upper middle income']
countries_list = [str(countries_frame[x][0]) for x in range(len(countries_frame)) if str(countries_frame[x][0]) not in exclusion_list]

country_dict = {}
for country in countries_list:
    filtered = df.filter(F.col('location') == country).filter(F.col('excess_mortality') != 0.0)
    if filtered.count() > 5:
        value = filtered.stat.corr("excess_mortality", "new_tests_smoothed")
    else:
        value = np.nan
    if not np.isnan(value):
        country_dict[country] = value

country_dict = dict(sorted(country_dict.items(), reverse=True, key=lambda item: item[1]))
print("As far as the correlation between new tests and excess mortality is concerned:\n")
print("The ten countries with the highest correlation are:")
for i, ct in enumerate(country_dict):
    if i == 10: break
    print(f"{ct}, with correlation equal to {country_dict[ct]:.3f}.")
country_dict = dict(sorted(country_dict.items(), reverse=False, key=lambda item: item[1]))
print("\nThe ten countries with the lowest correlation are:")
for i, ct in enumerate(country_dict):
    if i == 10: break
    print(f"{ct}, with correlation equal to {country_dict[ct]:.3f}.")

country_dict = {}
for country in countries_list:
    filtered = df.filter(F.col('location') == country).filter(F.col('excess_mortality') != 0.0)
    if filtered.count() > 5:
        value = filtered.stat.corr("excess_mortality", "total_vaccinations")
    else:
        value = np.nan
    if not np.isnan(value):
        country_dict[country] = value

country_dict = dict(sorted(country_dict.items(), reverse=True, key=lambda item: item[1]))
print("As far as the correlation between excess mortality and the course of the vaccinations is concerned:\n")
print("The ten countries with the highest correlation are:")
for i, ct in enumerate(country_dict):
    if i == 10: break
    print(f"{ct}, with correlation equal to {country_dict[ct]:.3f}.")
country_dict = dict(sorted(country_dict.items(), reverse=False, key=lambda item: item[1]))
print("\nThe ten countries with the lowest correlation are:")
for i, ct in enumerate(country_dict):
    if i == 10: break
    print(f"{ct}, with correlation equal to {country_dict[ct]:.3f}.")


#Covid and general health conditions on the country level
#Another interesting aspect of excess mortality is how it correlates with the general health conditions of a country's population. For this reason, we will first calculate the mean value of the 'female_smokers', 'male_smokers', 'diabetes_prevalence' and 'cardiovasc_death_rate' features, using the data on the last available date of our filtered DataFrame. Then, we will sort all countries with respect to their excess mortality per million, since a normalization is required when comparing different countries (and hence different populations). Finally, we will compare the values of the aforementioned features for the top 5 and the bottom 5 countries with their calculated mean values.

this_day = dates_list[-1]
filtered_df = df.filter(F.col('date') == this_day)

mean_fem_smokers = filtered_df.filter(F.col('female_smokers') != 0.0).select(F.mean(F.col('female_smokers'))).collect()[0][0]
mean_male_smokers = filtered_df.filter(F.col('male_smokers') != 0.0).select(F.mean(F.col('male_smokers'))).collect()[0][0]
mean_diabetes = filtered_df.filter(F.col('diabetes_prevalence') != 0.0).select(F.mean(F.col('diabetes_prevalence'))).collect()[0][0]
mean_card = filtered_df.filter(F.col('cardiovasc_death_rate') != 0.0).select(F.mean(F.col('cardiovasc_death_rate'))).collect()[0][0]

print(f'Based on data up to {this_day}, the mean percentage of female smokers is {mean_fem_smokers:.2f}%, while the corresponding number for male smokers is {mean_male_smokers:.2f}%.')
print(f'In addition, the mean percentage of people suffering from diabetes (aged 20-79) is {mean_diabetes:.2f}%, while the mean number of deaths per 100.000 people due to cardiovascular conditions is {mean_card:.2f}.')

filtered_df = filtered_df.filter(F.col('diabetes_prevalence') != 0.0).filter(F.col('cardiovasc_death_rate') != 0.0).filter(F.col('female_smokers') != 0.0).filter(F.col('male_smokers') != 0.0)
filtered_df.orderBy("excess_mortality_cumulative_per_million", ascending=False).select(["location", "excess_mortality_cumulative_per_million", "female_smokers", "male_smokers", "diabetes_prevalence", "cardiovasc_death_rate"]).toPandas().head(5)

filtered_df.orderBy("excess_mortality_cumulative_per_million", ascending=True).select(["location", "excess_mortality_cumulative_per_million", "female_smokers", "male_smokers", "diabetes_prevalence", "cardiovasc_death_rate"]).toPandas().head(5)

#k-Means Clustering
#Moving on to the final part of this EDA, we incorporate unsupervised learning methods, and more specifically k-Means clustering, in order to draw some additional information from our data. This is our final study on excess mortality and we intend to cluster countries together with respect to it, as well as the total number of cases - both normalized. This clustering will be performed on two different dates: the first and the final date present in our filtered DataFrame, in order to be able to see the evolution of the initial state. As previously done, we will only take into account countries with no missing values (i.e. zeroes) on excess mortality.

#As is always the case with k-Means clustering, the question that needs to be answered is "what is the optimal value of k?". While we could use methods such as the Elbow method to determine a good value for k, the purpose of this notebook is not an extensive study of clustering, but rather the presentation of a few basic methods for EDA using PySpark instead of widely used libraries such as Pandas or scikit-learn, for relatively small datasets. For this reason, we will simply create a scatterplot of the data and determine an optimal value for k through the visualization.

sns.set(style = "darkgrid")
    
fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(12,5))

for idx, (ax,this_day) in enumerate(zip([ax1,ax2],[exc_dates_list[0],exc_dates_list[-1]])):
    
    eff_df = df.filter(F.col('excess_mortality_cumulative_per_million') != 0.0).filter(F.col('date') == this_day).select(['total_cases_per_million','excess_mortality_cumulative_per_million','location'])

    pdf = eff_df.select(['total_cases_per_million','excess_mortality_cumulative_per_million']).toPandas()

    points = ax.scatter(pdf.total_cases_per_million, pdf.excess_mortality_cumulative_per_million,
                                  color=mycol, alpha=0.5)

    ax.set_title(f'Scatterplot of Countries as of {this_day}')
    ax.set_xlabel('Total Cases per Million')
    ax.set_ylabel('Excess Mortality (Cumulative) per Million')

plt.show()

matplotlib.rc_file_defaults()


from pyspark.ml.clustering import KMeans

sns.set(style = "darkgrid")

numclusters = [2,3]
colors = [mycol, mycomplcol, othercol1, othercol2, othercol3, othercol4]
    
fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(14,5))

for idx, (ax,this_day) in enumerate(zip([ax1,ax2],[exc_dates_list[0],exc_dates_list[-1]])):
    
    eff_df = df.filter(F.col('excess_mortality_cumulative_per_million') != 0.0).filter(F.col('date') == this_day).filter(F.col('date') == this_day).select(['total_cases_per_million','excess_mortality_cumulative_per_million','location'])

    vectorAssembler = VectorAssembler(inputCols = ['total_cases_per_million','excess_mortality_cumulative_per_million'], outputCol = "features")
    feat_df = vectorAssembler.transform(eff_df)
    feat_df = feat_df.select(['features','location'])

    kmeans = KMeans().setK(numclusters[idx]).setSeed(1).setFeaturesCol("features").setPredictionCol("cluster")
    model = kmeans.fit(feat_df)
    transformed = model.transform(feat_df)
    centroids = model.clusterCenters()

    transformed = transformed.join(eff_df, 'location')
    
    clusters, centers, images = {}, {}, {}

    for i in range(numclusters[idx]):

        clusters[i] = transformed.filter(F.col('cluster')==i).select(['location','cluster','total_cases_per_million',
                                                              'excess_mortality_cumulative_per_million']).toPandas().set_index('location')

        images[i] = ax.scatter(clusters[i].total_cases_per_million, clusters[i].excess_mortality_cumulative_per_million,
                                  color=colors[i], alpha=0.5)
        centers[i] = ax.scatter(centroids[i][0], centroids[i][1], color=colors[i], marker='x')

    clusttuple = (images[i] for i in range(numclusters[idx]))
    clustnames = ('Cluster '+str(i+1) for i in range(numclusters[idx]))

    ax.legend(clusttuple, clustnames, loc='best')

    ax.set_title(f'Clusters of Countries as of {this_day}')
    ax.set_xlabel('Total Cases per Million')
    ax.set_ylabel('Excess Mortality (Cumulative) per Million')

plt.show()

matplotlib.rc_file_defaults()

print(*clusters[2].index, sep=', ')

sns.set(style = "darkgrid")
    
fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(14,5))

for idx, (ax,this_day) in enumerate(zip([ax1,ax2],[dates_list[0],dates_list[-1]])):
    
    eff_df = df.filter(F.col('human_development_index') != 0.0).filter(F.col('reproduction_rate') != 0.0).filter(F.col('date') == this_day).select(['human_development_index','reproduction_rate','location'])

    pdf = eff_df.select(['human_development_index','reproduction_rate']).toPandas()

    points = ax.scatter(pdf.human_development_index, pdf.reproduction_rate, color=mycol, alpha=0.5)

    ax.set_title(f'Scatterplot of Countries as of {this_day}')
    ax.set_yticks([0.0,0.5,1.0,1.5,2.0])
    ax.set_xlabel('Human Development Index')
    ax.set_ylabel('Reproduction Rate')

plt.show()




sns.set(style = "darkgrid")

numclusters = [3,3]
colors = [mycol, mycomplcol, othercol1, othercol2, othercol3, othercol4]
    
fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(14,5))

for idx, (ax,this_day) in enumerate(zip([ax1,ax2],[dates_list[0],dates_list[-1]])):
    
    eff_df = df.filter(F.col('human_development_index') != 0.0).filter(F.col('reproduction_rate') != 0.0).filter(F.col('date') == this_day).select(['human_development_index','reproduction_rate','location'])

    vectorAssembler = VectorAssembler(inputCols = ['human_development_index','reproduction_rate'], outputCol = "features")
    feat_df = vectorAssembler.transform(eff_df)
    feat_df = feat_df.select(['features','location'])

    kmeans = KMeans().setK(numclusters[idx]).setSeed(1).setFeaturesCol("features").setPredictionCol("cluster")
    model = kmeans.fit(feat_df)
    transformed = model.transform(feat_df)
    centroids = model.clusterCenters()

    transformed = transformed.join(eff_df, 'location')
    
    clusters, centers, images = {}, {}, {}

    for i in range(numclusters[idx]):

        clusters[i] = transformed.filter(F.col('cluster')==i).select(['location','cluster','reproduction_rate',
                                                              'human_development_index']).toPandas().set_index('location')

        images[i] = ax.scatter(clusters[i].human_development_index, clusters[i].reproduction_rate, color=colors[i], alpha=0.5)
        centers[i] = ax.scatter(centroids[i][0], centroids[i][1], color=colors[i], marker='x')

    clusttuple = (images[i] for i in range(numclusters[idx]))
    clustnames = ('Cluster '+str(i+1) for i in range(numclusters[idx]))

    ax.legend(clusttuple, clustnames, loc='best')

    ax.set_title(f'Clusters of Countries as of {this_day}')
    ax.set_yticks([0.0,0.5,1.0,1.5,2.0])
    ax.set_xlabel('Human Development Index')
    ax.set_ylabel('Reproduction Rate')

plt.show()

matplotlib.rc_file_defaults()

matplotlib.rc_file_defaults()


