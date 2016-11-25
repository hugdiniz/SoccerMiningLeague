SELECT match_api_id, country_id, league_id, stage, date, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal, home_player_X1, home_player_X2, home_player_X3, home_player_X4, home_player_X5, home_player_X6, home_player_X7, home_player_X8, home_player_X9, home_player_X10, home_player_X11, away_player_X1, away_player_X2, away_player_X3, away_player_X4, away_player_X5, away_player_X6, away_player_X7, away_player_X8, away_player_X9, away_player_X10, away_player_X11, home_player_Y1, home_player_Y2, home_player_Y3, home_player_Y4, home_player_Y5, home_player_Y6, home_player_Y7, home_player_Y8, home_player_Y9, home_player_Y10, home_player_Y11, away_player_Y1, away_player_Y2, away_player_Y3, away_player_Y4, away_player_Y5, away_player_Y6, away_player_Y7, away_player_Y8, away_player_Y9, away_player_Y10, away_player_Y11, home_player_1, home_player_2, home_player_3, home_player_4, home_player_5, home_player_6, home_player_7, home_player_8, home_player_9, home_player_10, home_player_11, away_player_1, away_player_2, away_player_3, away_player_4, away_player_5, away_player_6, away_player_7, away_player_8, away_player_9, away_player_10, away_player_11 

FROM [Match] 
JOIN v_player_attributes_recents as ph1 on home_player_1 = ph1.player_api_id 
JOIN v_player_attributes_recents as ph2 on home_player_2 = ph2.player_api_id
JOIN v_player_attributes_recents as ph3 on home_player_3 = ph3.player_api_id 
JOIN v_player_attributes_recents as ph4 on home_player_4 = ph4.player_api_id 
JOIN v_player_attributes_recents as ph5 on home_player_5 = ph5.player_api_id 
JOIN v_player_attributes_recents as ph6 on home_player_6 = ph6.player_api_id 
JOIN v_player_attributes_recents as ph7 on home_player_7 = ph7.player_api_id 
JOIN v_player_attributes_recents as ph8 on home_player_8 = ph8.player_api_id 
JOIN v_player_attributes_recents as ph9 on home_player_9 = ph9.player_api_id 
JOIN v_player_attributes_recents as ph10 on home_player_10 = ph10.player_api_id 
JOIN v_player_attributes_recents as ph11 on home_player_11 = ph11.player_api_id 

JOIN v_player_attributes_recents as pa1 on away_player_1 = pa1.player_api_id 
JOIN v_player_attributes_recents as pa2 on away_player_2 = pa2.player_api_id
JOIN v_player_attributes_recents as pa3 on away_player_3 = pa3.player_api_id 
JOIN v_player_attributes_recents as pa4 on away_player_4 = pa4.player_api_id 
JOIN v_player_attributes_recents as pa5 on away_player_5 = pa5.player_api_id 
JOIN v_player_attributes_recents as pa6 on away_player_6 = pa6.player_api_id 
JOIN v_player_attributes_recents as pa7 on away_player_7 = pa7.player_api_id 
JOIN v_player_attributes_recents as pa8 on away_player_8 = pa8.player_api_id 
JOIN v_player_attributes_recents as pa9 on away_player_9 = pa9.player_api_id 
JOIN v_player_attributes_recents as pa10 on away_player_10 = pa10.player_api_id 
JOIN v_player_attributes_recents as pa11 on away_player_11 = pa11.player_api_id 

JOIN Team_Attributes  as tah on home_team_api_id = tah.team_api_id
JOIN Team_Attributes  as taa on away_team_api_id = taa.team_api_id

WHERE season like '2015/2016' AND league_id = 7809 AND taa.date = '2015-09-10 00:00:00' AND tah.date = '2015-09-10 00:00:00'
;
