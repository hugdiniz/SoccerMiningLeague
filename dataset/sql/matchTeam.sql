SELECT mat.match_api_id, mat.country_id, mat.league_id, mat.stage, mat.date, mat.home_team_api_id, mat.away_team_api_id, mat.home_team_goal, mat.away_team_goal, mat.home_player_X1, mat.home_player_X2, mat.home_player_X3, mat.home_player_X4, mat.home_player_X5, mat.home_player_X6, mat.home_player_X7, mat.home_player_X8, mat.home_player_X9, mat.home_player_X10, mat.home_player_X11, mat.away_player_X1, mat.away_player_X2, mat.away_player_X3, mat.away_player_X4, mat.away_player_X5, mat.away_player_X6, mat.away_player_X7, mat.away_player_X8, mat.away_player_X9, mat.away_player_X10, mat.away_player_X11, mat.home_player_Y1, mat.home_player_Y2, mat.home_player_Y3, mat.home_player_Y4, mat.home_player_Y5, mat.home_player_Y6, mat.home_player_Y7, mat.home_player_Y8, mat.home_player_Y9, mat.home_player_Y10, mat.home_player_Y11, mat.away_player_Y1, mat.away_player_Y2, mat.away_player_Y3, mat.away_player_Y4, mat.away_player_Y5, mat.away_player_Y6, mat.away_player_Y7, mat.away_player_Y8, mat.away_player_Y9, mat.away_player_Y10, mat.away_player_Y11, mat.home_player_1, mat.home_player_2, mat.home_player_3, mat.home_player_4, mat.home_player_5, mat.home_player_6, mat.home_player_7, mat.home_player_8, mat.home_player_9, mat.home_player_10, mat.home_player_11, mat.away_player_1, mat.away_player_2, mat.away_player_3, mat.away_player_4, mat.away_player_5, mat.away_player_6, mat.away_player_7, mat.away_player_8, mat.away_player_9, mat.away_player_10, mat.away_player_11,
tah.buildUpPlaySpeed, tah.buildUpPlayDribbling, tah.buildUpPlayPassing, tah.buildUpPlayPositioningClass, tah.chanceCreationPassing, tah.chanceCreationCrossing, tah.chanceCreationShooting, tah.chanceCreationPositioningClass, tah.defencePressure, tah.defenceAggression, tah.defenceTeamWidth, tah.defenceDefenderLineClass, taa.buildUpPlaySpeed, taa.buildUpPlayDribbling, taa.buildUpPlayPassing, taa.buildUpPlayPositioningClass, taa.chanceCreationPassing, taa.chanceCreationCrossing, taa.chanceCreationShooting, taa.chanceCreationPositioningClass, taa.defencePressure, taa.defenceAggression, taa.defenceTeamWidth, taa.defenceDefenderLineClass       

FROM Match as mat

JOIN Team_Attributes  as tah on home_team_api_id = tah.team_api_id
JOIN Team_Attributes  as taa on away_team_api_id = taa.team_api_id

WHERE season like '2015/2016' AND league_id = 7809 AND taa.date = '2015-09-10 00:00:00' AND tah.date = '2015-09-10 00:00:00'
;
