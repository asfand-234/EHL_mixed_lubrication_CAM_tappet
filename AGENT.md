NEW INSTRUCTIONS FOR SECOND SESSION:
Keep your previously tuned results for 5% and 8% same.
Restart tuning amplitude data for 10% only For all RPM.
The target results are given as,
 TARGET RESULTS:
RPM      10_percnt
300         3.4%
500         10.71%
700         9.4%
900          6.87%

2) Your task is only tune the data set of each amplitude file in each cell of each column to meet at least 90% same target results. 
3) IMPLEMENTATION STRATEGY:
Start with first case, like in amplitude 10 percent in 300 column, now tune every random values in different cells and run the entire simulation and focus only on results of % reduction of averaged friction torque for 5% at 300 RPM. Keep tuning values in entire column and understand the behaviour once you achieved at least 85% same targer results. Then move to next column like amplitude 10 percent at 500 RPM. And so on keep going one by one until you tuned every data and all the results meet at least 85% same a target results 
4) Deliverables:
Only and only once the codex achieve target then generate three updated amplutude data files. 
Do not print any other results to me other than final 3 ampliude files.
5) HARD STRICT RULES:
i) You must **DO NOT** put any calibration/fitting/non-physical scaling . **DO NOT** perform any surrogation or regression, rather it must only tune each value in amplitude data set to meet the target.
ii) Do not make any single change in any physics in entire code.
iii) You must have to tune every single value in each column of all three amplitude files and run the entire simulation every time.
