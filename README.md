# nrl_analysis
Statistical Analysis of the NRL

In the following project I use the following tools and python packages in order to analyse the NRL:
- Optimus: I use this python package to clean the dataset found on this website http://www.aussportsbetting.com/data/historical-nrl-results-and-odds-data/
- Pandas: This well known package was used to do additional data cleaning that I could not achieve using Optimus. For example, in order to find a particular team's position on the ladder before a particular match, I had to use a for loop to iterate through each row.
- H2O: This is a machine learning library that I used to create a multinomial regression model, as well as a grid search of multiple gradient boosting models with different sets of parameters.
- mlflow: This is a library which helps you record the key metrics for each model, as well as the artifacts of each model (such as the model itself, variable importance plots, etc)
- AWS s3: To back up metrics and artifacts. 
- Selenium: To scrape odds and fixtures from different websites. (The scripts that scrape the odds and fixtures are currently in .gitignore as I am waiting for confirmation from the company to see if I can make this public)
- Airflow: Created a simple dag which firstly runs the web scrapers, it then runs the script which concats the next fixtures + odds together before finishing with the script which actually forms the predictions using the mojo file.


Thanks to Australia Sports Betting for providing the inital dataset and also to Poinstbet for providing future odds and games.
