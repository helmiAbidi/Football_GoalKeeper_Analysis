import pandas as pd
import numpy as np
import datetime
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from functools import reduce
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
###############################################################
######################## DATA CLEANING ########################
###############################################################
def rename_FBREF_columns(df):
    ## Clean Columns 
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    for col in df.columns.values:
        if 'Unnamed' in col:
            new_col = col[col.rindex('_')+1:]
            df.rename(columns = {col:new_col}, inplace = True)
    return df

def calculate_saving_ability_score(df,season):
    '''
    
    + ((df['Penalty Kicks_PKsv']-min_penalty_sv)*0.5/(max_penalty_sv-min_penalty_sv)+ df['Penalty Kicks_PKsv']/df['Penalty Kicks_PKatt']*0.5 )*0.1
    '''
    min_psxg, max_psxg = df['Expected_PSxG+/-'].min(),df['Expected_PSxG+/-'].max()
    #min_penalty_sv, max_penalty_sv = df['Penalty Kicks_PKsv'].min(),df['Penalty Kicks_PKsv'].max()

    df["psxg_score"] = 0
    df.loc[df['Expected_PSxG+/-']>0,"psxg_score"] = df['Expected_PSxG+/-']*0.7/max_psxg + 0.3
    df.loc[df['Expected_PSxG+/-']<0,"psxg_score"] = df['Expected_PSxG+/-']/min_psxg*0.2 + 0.1

    df['saving_ability_score_'+ season] = df['Performance_CS%']/100 * 0.3 
    +(df['Performance_Save%']*0.5/100 +  df["psxg_score"]*0.5) * 0.7

    #df['saving_ability_score_'+ season] = df['saving_ability_score_'+ season] * df['Comp_Score'] /100

    min_saving_score, max_saving_score = df['saving_ability_score_'+ season].min(),df['saving_ability_score_'+ season].max() 
     

    df['saving_ability_score_'+ season] = ((df['saving_ability_score_'+ season]-min_saving_score)/(max_saving_score-min_saving_score)) *60+35 
    
    return df

def calculate_distribution_score(df, season):
    max_avg_len, min_avg_len = df['Goal Kicks_AvgLen'].max(),  df['Goal Kicks_AvgLen'].min()
    df['Distribution_Score_' + season] = df['Launched_Cmp%']* 0.5/100+\
                                        (df['Goal Kicks_Launch%'] * 0.5/100 + (df['Goal Kicks_AvgLen']-min_avg_len)*0.5/(max_avg_len-min_avg_len)) * 0.5
                                        
    #df['Distribution_Score_' + season] = df['Distribution_Score_'+ season] * df['Comp_Score'] /100
    min_dist_score, max_dist_score = df['Distribution_Score_' + season].min(),df['Distribution_Score_' + season].max() 
    df['Distribution_Score_' + season] = ((df['Distribution_Score_' + season]-min_dist_score)/(max_dist_score-min_dist_score)) *60+35 

    return df

def calculate_ball_quality_score(df,season):
    min_pass_att, max_pass_att = df['Passes_Att'].min(),df['Passes_Att'].max()
    
    df['Ball_Quality_Score_' + season] = (df['Passes_Att']-min_pass_att)/(max_pass_att-min_pass_att)

    #df['Ball_Quality_Score_' + season] = df['Ball_Quality_Score_'+ season] * df['Comp_Score'] /100
    min_ball_score, max_ball_score = df['Ball_Quality_Score_' + season].min(),df['Ball_Quality_Score_' + season].max() 
    df['Ball_Quality_Score_' + season] = ((df['Ball_Quality_Score_' + season]-min_ball_score)/(max_ball_score-min_ball_score)) *60+35  
      
    return df

def calculate_positioning_score(df,season):
    min_goal_fk, max_goal_fk = df['Goals_FK'].min(), df['Goals_FK'].max()
    min_goal_ck, max_goal_ck = df['Goals_CK'].min(), df['Goals_CK'].max()
    min_crosses, max_crosses = df['Crosses_Opp'].min(), df['Crosses_Opp'].max()
    df['Positioning_Score_' + season] = (1-(df['Goals_FK']-min_goal_fk)*0.5/(max_goal_fk-min_goal_fk)) * 0.25 + \
                                        (1-(df['Goals_CK']-min_goal_ck)*0.5/(max_goal_ck-min_goal_ck) )* 0.25 + \
                                        ((df['Crosses_Opp']-min_crosses)*0.5/(max_crosses-min_crosses) + df['Crosses_Stp%']*0.5/100) * 0.5   
                                        
    #df['Positioning_Score_' + season] = df['Positioning_Score_'+ season] * df['Comp_Score'] /100
    min_pos_score, max_pos_score = df['Positioning_Score_' + season].min(),df['Positioning_Score_' + season].max() 
    df['Positioning_Score_' + season] = ((df['Positioning_Score_' + season]-min_pos_score)/(max_pos_score-min_pos_score)) *60+35 

    return df

def calculate_sweeper_score(df,season):
    min_sweeper_opa, max_sweeper_opa= df['Sweeper_#OPA/90'].min(),df['Sweeper_#OPA/90'].max()
    min_sweeper_avg_len, max_sweeper_avg_len = df['Sweeper_AvgDist'].min(),df['Sweeper_AvgDist'].max()


    df['Sweeper_Score_'+season] = (df['Sweeper_#OPA/90']-min_sweeper_opa)*0.5/(max_sweeper_opa-min_sweeper_opa)+\
                                            (df['Sweeper_AvgDist']-min_sweeper_avg_len)*0.5/(max_sweeper_avg_len-min_sweeper_avg_len)

    #df['Sweeper_Score_' + season] = df['Sweeper_Score_'+ season] * df['Comp_Score'] /100
    min_sweeper_score, max_sweeper_score = df['Sweeper_Score_'+season].min(),df['Sweeper_Score_'+season].max() 

    df['Sweeper_Score_'+season] = ((df['Sweeper_Score_'+season]-min_sweeper_score)/(max_sweeper_score-min_sweeper_score)) *60+35     
    return df

def calculate_comp_score(df):
    df['Comp_Score'] = 0
    df.loc[df['Comp']== 'eng Premier League','Comp_Score'] = 100
    df.loc[df['Comp']== 'es La Liga','Comp_Score'] = 80
    df.loc[df['Comp']== 'it Serie A','Comp_Score'] = 60
    df.loc[df['Comp']== 'de Bundesliga','Comp_Score'] = 50
    df.loc[df['Comp']== 'fr Ligue 1','Comp_Score'] = 45
    return df


def prepare_one_season_FBREF_data(season):
    ### Read Data 
    standard_stats = pd.read_csv('./data/player_stats/season_'+season+'/player_stats_'+season+'_standard_stats.csv',header=[0,1])
    gk_stats = pd.read_csv('./data/player_stats/season_'+season+'/player_stats_'+ season +'_gk_stats.csv',header=[0,1])
    gk_advanced_stats = pd.read_csv('./data/player_stats/season_'+season+'/player_stats_'+season + '_advanced_gk_stats.csv',header=[0,1])
    
    ### clean column names 
    ################### gk_standard #########################
    standard_stats = rename_FBREF_columns(standard_stats)
    relevant_columns = ['Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Age','Born','Performance_CrdY', 'Performance_CrdR']

    gk_standard_df = standard_stats[standard_stats['Pos']=='GK']
    gk_standard_df = gk_standard_df[relevant_columns]
    ################### gk_stats ############################
    gk_stats = rename_FBREF_columns(gk_stats)
    relevant_columns = ['Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Age',
                        'Playing Time_MP','Playing Time_Min', ## playing time 
                        'Performance_GA', 'Performance_GA90','Performance_Save%',
                        'Performance_CS','Performance_CS%','Penalty Kicks_PKatt', 'Penalty Kicks_PKA',
                        'Penalty Kicks_PKsv', 'Penalty Kicks_PKm'] ## saving ability
    gk_stats = gk_stats[relevant_columns]

    ################### gk_advanced_stats ############################
    gk_advanced_stats = rename_FBREF_columns(gk_advanced_stats)
    relevant_columns = ['Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Age',
                        'Goals_FK', 'Goals_CK','Expected_PSxG','Expected_PSxG/SoT','Expected_PSxG+/-','Expected_/90', ## saving ability
                        'Launched_Att', 'Launched_Cmp%','Goal Kicks_Att', 'Goal Kicks_Launch%', 'Goal Kicks_AvgLen' , ## goal distribution length and accuracy
                        'Passes_Att','Passes_Launch%', 'Passes_AvgLen','Sweeper_#OPA/90', 'Sweeper_AvgDist', ## quality with the ball 
                        'Crosses_Opp','Crosses_Stp%'] ## positioning ability   
    gk_advanced_stats = gk_advanced_stats[relevant_columns]
    
    ### Merge all gk stats
    gk_all_stats = pd.merge(gk_stats,gk_standard_df,how="inner",
                  on=['Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Age'])

    gk_all_stats = pd.merge(gk_all_stats,gk_advanced_stats,how="inner",
                    on=['Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Age'])

    gk_all_stats = gk_all_stats[(gk_all_stats['Playing Time_MP']>10)&(gk_all_stats['Playing Time_Min']>450)]
    
    non_nan_columns = ['Goals_FK', 'Goals_CK','Expected_PSxG','Expected_PSxG/SoT','Expected_PSxG+/-','Expected_/90', ## saving ability
                        'Launched_Att', 'Launched_Cmp%','Goal Kicks_Att', 'Goal Kicks_Launch%', 'Goal Kicks_AvgLen' , ## goal distribution length and accuracy
                        'Passes_Att','Passes_Launch%', 'Passes_AvgLen','Sweeper_#OPA/90', 'Sweeper_AvgDist', ## quality with the ball 
                        'Crosses_Opp','Crosses_Stp%']
    gk_all_stats.dropna(axis=0,subset= non_nan_columns, inplace=True)
    ### create match based features
    gk_all_stats = calculate_comp_score(gk_all_stats)
    gk_all_stats = calculate_saving_ability_score(gk_all_stats,season)
    gk_all_stats = calculate_distribution_score(gk_all_stats,season)
    gk_all_stats = calculate_ball_quality_score(gk_all_stats,season)
    gk_all_stats = calculate_positioning_score(gk_all_stats,season)
    gk_all_stats = calculate_sweeper_score(gk_all_stats,season)
    
    
    
    return gk_all_stats


def prepare_one_season_FIFA_data(season):
    fifa_ds = pd.read_csv('../data/FIFA_DS/players_'+season+'.csv')
    gk_fifa_ds = fifa_ds[fifa_ds['player_positions']=="GK"]
    relevant_leagues = ['Spain Primera Division', 'German 1. Bundesliga','English Premier League', 'French Ligue 1', 'Italian Serie A']
    #gk_fifa_ds = gk_fifa_ds[gk_fifa_ds['league_name'].isin(relevant_leagues)]
    
    gk_fifa_ds['league_name'].replace('Spain Primera Division','es La Liga',inplace=True)
    gk_fifa_ds['league_name'].replace('German 1. Bundesliga','de Bundesliga',inplace=True)
    gk_fifa_ds['league_name'].replace('English Premier League','eng Premier League',inplace=True)
    gk_fifa_ds['league_name'].replace('French Ligue 1','fr Ligue 1',inplace=True)
    gk_fifa_ds['league_name'].replace('Italian Serie A','it Serie A',inplace=True)

    print(gk_fifa_ds['league_name'].unique())
    
    fifa_gk_columns = ['long_name','player_positions','overall', 'potential', 'value_eur', 'wage_eur',\
                   'club_name','age','dob','height_cm','weight_kg','league_name',\
                    'skill_long_passing','skill_ball_control',\
                    'power_strength','mentality_vision',\
                    'mentality_composure','goalkeeping_diving','goalkeeping_handling',\
                    'goalkeeping_kicking','goalkeeping_positioning','goalkeeping_reflexes','goalkeeping_speed']
    
    gk_fifa_ds= gk_fifa_ds[fifa_gk_columns]
    column_dict = {}
    for column in fifa_gk_columns:
        if column in ['long_name','dob']:
            column_dict[column] = column
        else:
            column_dict[column] = column + "_" + season
    gk_fifa_ds.rename(columns=column_dict,inplace=True)
    #gk_fifa_ds['birth_year'] = gk_fifa_ds['dob'].apply(lambda x: int(x[:x.index('-')]))
    return gk_fifa_ds

def merge_datasets(fbref_ds,fifa_ds):
    
    fbref_ds['short_version'] = fbref_ds['Player'] 
    fbref_ds['short_version'][fbref_ds['short_version'].str.contains(" ")] = fbref_ds[fbref_ds['short_version'].str.contains(" ")]['short_version'].apply(lambda x: x[0] + ". " + x[x.index(' ')+1:])

    gk_merged_stats = pd.merge(fbref_ds,fifa_ds, left_on=['short_version', 'Born', 'Comp'], right_on=['short_name','birth_year','league_name'],how='inner')
    
    return gk_merged_stats

def create_gk_features(gk_merged_stats,season):
    ## chosse intuitively relevant features and check their corrolation with the overall value
    relevant_columns = ['Age','height_cm', 'weight_kg',
                        'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking','goalkeeping_positioning',
                        'goalkeeping_reflexes', 'goalkeeping_speed','overall']
    gk_features = gk_merged_stats[relevant_columns]

    gk_features['mental_score'] = gk_merged_stats[['mentality_aggression', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure']].mean(axis=1)
   
    return gk_features


###############################################################
######################## DATA Analysis ########################
###############################################################

def classify_df(n_clusters, df, feature_columns):
    
    all_features = df[feature_columns]

    all_features_array = all_features.to_numpy(dtype=float)
    classifier = AgglomerativeClustering(n_clusters=n_clusters).fit(all_features_array)
    return classifier.labels_

def visualize_clusters(labels,df,x_label, y_label):
    fig = plt.figure(figsize=(30,10))
    ## pick 10 random players. 
    colors = ['b','r','g'] # ,'k','c'
    for l in range(3):
        names, gk_scores = [],[]
        label_indices = np.where(labels==l)[0].tolist()
        label_indices = label_indices[:5]
        ## I need names and GK_Score
        for label_index in label_indices:
            try:
                name = df.loc[label_index,x_label]
                names.append(name[name.index(' ')+1:])
                gk_scores.append(df.loc[label_index,y_label])
            except:
                continue
        plt.scatter(names, gk_scores, c=colors[l])

    plt.show() 

def visualize_simple_radar_plot(player_name, df,radar_columns,theta_columns):
    radar_df = df[df["Player"]==player_name]
    
    r = [0,10,20,30,40,50,60,70,80,90,100]
    radar_df = radar_df[radar_columns]
    
    test_df = pd.DataFrame(dict(
        r= radar_df.iloc[0].values.tolist(),
        theta= theta_columns))
    fig = px.line_polar(test_df, r='r', theta='theta', line_close=True)
    fig.show()
    
def compare_two_players(player_name_1,player_name_2,radar_columns, categories,df,title):
    categories = [*categories, categories[0]]

    player_1 = [*df.loc[df['Player']== player_name_1,radar_columns].values.flatten().tolist()]
    player_2 = [*df.loc[df['Player']== player_name_2,radar_columns].values.flatten().tolist()]
    player_1 = [*player_1, player_1[0]]
    player_2 = [*player_2, player_2[0]]


    fig = go.Figure(
        data=[
            go.Scatterpolar(r=player_1, theta=categories, fill='toself', name=player_name_1),
            go.Scatterpolar(r=player_2, theta=categories, fill='toself', name=player_name_2)
        ],
        layout=go.Layout(
            title=go.layout.Title(text=title),
            polar={'radialaxis': {'visible': True}},
            showlegend=True
        )
    )

    pyo.plot(fig)
    return

def prepare_train_datasets(data_frames, target_season,history):

    gk_fifa_ds_train = reduce(lambda  left,right: pd.merge(left,right,on=['long_name','dob'],
                                            how='inner'), data_frames)
    
    if history ==2:
        feature_columns = ['overall_19', 'goalkeeping_reflexes_19', 'goalkeeping_diving_19',
                            'goalkeeping_positioning_19', 'goalkeeping_handling_19', 'overall_18',
                            'goalkeeping_reflexes_18', 'goalkeeping_diving_18', 'goalkeeping_positioning_18',
                            'goalkeeping_handling_18', 'goalkeeping_kicking_19', 'goalkeeping_kicking_18']
    elif history ==4:
        feature_columns = ['overall_16','goalkeeping_diving_16', 'goalkeeping_handling_16', 'goalkeeping_kicking_16', 'goalkeeping_positioning_16', 'goalkeeping_reflexes_16', 'goalkeeping_speed_16',
                    'overall_17', 'goalkeeping_diving_17', 'goalkeeping_handling_17', 'goalkeeping_kicking_17', 'goalkeeping_positioning_17', 'goalkeeping_reflexes_17', 'goalkeeping_speed_17',
                        'overall_18','goalkeeping_diving_18', 'goalkeeping_handling_18', 'goalkeeping_kicking_18', 'goalkeeping_positioning_18', 'goalkeeping_reflexes_18', 'goalkeeping_speed_18',
                        'overall_19','goalkeeping_diving_19', 'goalkeeping_handling_19', 'goalkeeping_kicking_19', 'goalkeeping_positioning_19', 'goalkeeping_reflexes_19', 'goalkeeping_speed_19']
    
    new_feature_columns = []
    for feature in feature_columns:
        season = feature[feature.rindex('_')+1:]
        if season == "16":
            new_feature_columns.append(feature[:feature.rindex('_')+1] + str(target_season-4))
        elif season == "17":
            new_feature_columns.append(feature[:feature.rindex('_')+1] + str(target_season-3))
        elif season == "18":
            new_feature_columns.append(feature[:feature.rindex('_')+1] + str(target_season-2))
        elif season == "19":
            new_feature_columns.append(feature[:feature.rindex('_')+1] + str(target_season-1))
    
    gk_features = gk_fifa_ds_train[new_feature_columns]
    target_train = gk_fifa_ds_train['overall_' + str(target_season)]
    
    return [gk_features,target_train] #.to_numpy(dtype=float)

def prepare_overall_dataset(overalls,season):
    gk_overalls = reduce(lambda  left,right: pd.merge(left,right,on=['long_name','dob'],
                                            how='right'), overalls)

    gk_overalls = gk_overalls[gk_overalls.isnull().sum(axis=1) < 3]
    #gk_overalls = gk_overalls.fillna(gk_overalls.mean(axis=1), axis=1)
    gk_overalls = gk_overalls.T.fillna(gk_overalls.iloc[:,2:6].min(axis=1)).T
    
    feature_columns = ['overall_'+ str(season-4),'overall_'+ str(season-3),'overall_'+ str(season-2),'overall_'+ str(season-1)]
    gk_features = gk_overalls[feature_columns]
    target_train = gk_overalls['overall_' + str(season)]
    
    return  [gk_features.to_numpy(dtype=float),target_train.to_numpy(dtype=float)]

def baseline_model(feature_df,target,season):
    feature_df['diff_overall'] = feature_df['overall_' + str(season-1)] - feature_df['overall_' + str(season-2)]

    feature_df['diff_reflexes'] = feature_df['goalkeeping_reflexes_'+ str(season-1)] - feature_df['goalkeeping_reflexes_'+ str(season-2)]
    feature_df['diff_diving']= feature_df['goalkeeping_diving_'+ str(season-1)] - feature_df['goalkeeping_diving_'+ str(season-2)]
    feature_df['diff_positioning']= feature_df['goalkeeping_positioning_'+ str(season-1)] - feature_df['goalkeeping_positioning_'+ str(season-2)]
    feature_df['diff_handling']= feature_df['goalkeeping_handling_'+ str(season-1)] - feature_df['goalkeeping_handling_'+ str(season-2)]
    feature_df['diff_kicking']= feature_df['goalkeeping_kicking_'+ str(season-1)] - feature_df['goalkeeping_kicking_'+ str(season-2)]

    feature_df['diff_features'] = feature_df[['diff_reflexes','diff_diving','diff_positioning','diff_handling','diff_kicking']].mean(axis=1)
    feature_df['predicted_overall'] = feature_df['overall_'+ str(season-1)] + feature_df[['diff_overall','diff_features']].mean(axis=1)

    predicted = feature_df['predicted_overall'].to_numpy(dtype=float)
    target = target.to_numpy(dtype=float)
    return [predicted,target]

def evaluate_regression_model(model, test_data,test_target):
    predicted_overall = model.predict(test_data)
    predicted_overall = np.round(predicted_overall)

    return [mean_squared_error(test_target, predicted_overall),mean_absolute_error(test_target, predicted_overall)]

def progress_type_accuracy(model, test_data, test_target, old_overall):
    predicted_overall = model.predict(test_data)
    predicted_overall = np.round(predicted_overall)


    train_diff = predicted_overall - old_overall
    test_diff = test_target - old_overall


    train_diff_sign = np.sign(train_diff)
    test_diff_sign = np.sign(test_diff)

    test_sign = train_diff_sign == test_diff_sign

     
    return np.count_nonzero(test_sign==True)/ test_sign.shape[0] 