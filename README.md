# mini-salary-predictor
A mini project that goes over basic web scraping methods and utilizing ensemble machine learning modeling methods as well as basic natural language processing methods to predict factors that affect UK Data Scientist salaries.


**Objective:**

The goal is to identify, if any, notable factors that can help in predicting your potential salary for jobs in the data science and related fields in various cities in the UK. The website of choice was indeed.co.uk, a well-known job search website that provides most of the necessary information that can help us model and predict. The first step would be to gather the data from Indeed.

**Step 1: Web Scraping**

To crawl a third party website is not as easy as it seems. You have to set up your crawler in such that the servers hosting the website does not get overloaded with requests and detect your web scraping, if so they can block/hinder your efforts by placing a captcha page or block your IP address entirely.

To do this, you have to space out your crawlers, or in other words, do it city by city, and set a random time gap between each page scraped. With that in mind, your crawler is done with a Python module called BeautifulSoup, one that allows you to note specific HTML tags from the website (when using an element inspection) and crawl out useful text information. With this, you can retrieve key points like the job title, company name, location, city and the salaries provided, and store them in a python dictionary of lists, which is what I’ve done. Saving this as a function allows you to repeat this method per city with little trouble. Note however that the website may still detect your crawling, so this step would have to be tinkered with in future efforts.

**Step 2: Placing data in a tabular format, cleaning & saving as a .csv**

 With each city saved in a dictionary, it is very easy to convert them into a DataFrame, where each can then be merged into a larger, master DataFrame with all cities. Your next step will be then to clean your DataFrame. Look out for:
- Duplicate values
- Job listings with no salaries provided
- Formatting of your salary column (for indeed this is a text range that python can’t read numerically) e.g. (‘£25,000 - £30,000’)

After removing duplicate values and jobs with no listed salaries from your data, the trickiest step here is to convert these text salary ranges into a numerical format. To do this, python has a multiple methods that can:
- Split the texts into 2 items
- Remove unnecessary characters like ‘,’ or ‘-’
- Convert the text digits into numbers
- Apply an average function that takes the mean of the lower & upper value

With all of that done as a function, you now have your salaries in a readable format, ready for modeling. 

**Step 3: Modeling**

Firstly, you have to decide what value you want to constitute a ‘high’ salary. In this case, I went with anything above the 75th percentile (£40,000). Python again has a function that can read your salary column, and create a new one with binarized values (0 & 1) that can tell you if a salary is high or not. This will be the target variable for your modeling.

Unfortunately for this dataset, this is where it gets a bit strange. Generally you want your models to outperform your baseline score for them to be of any use. In our case the baseline score is calculated by the majority class when we split the high & low salaries, so in this case a 0.75 score. The first model tries to identify which city (out of 10) pays the most. After dummying the city column and only using this in the model as the predictor variable, a few models were used to test this out. They include:
- Logistic Regression
- KNN
- Decision Trees
- Random Forests
- AdaBoost
- XGBoost

The results weren’t great. All of these models had very similar scores (~0.7 average cross_validated scores) and aren’t better than the baseline score. It appears that for this dataset at least, what city you were in does not affect significantly on whether you’re paid a higher salary or not. Further inspection with looking for the key coefficients found no significant differences between cities in affecting salaries.

Further models are implemented on other features in the dataset, for instance identifying certain words in the job title or what company is listing the job. These are done with natural language processing methods (nlp), that can filter out each word in a text column and assign a numeric value to it. I implemented 2 nlp methods, one called Count Vectorizer (cvec) and another, supposedly better method called a TF-IDF Vectorizer (tvec). Both filter out commonly used English words and hope to identify key terms from the columns. 

Similar models to the above for cities are ran for both the cover & tvec methods and unfortunately like the city models, we can’t find any significant factors. Any key terms like ‘senior’ or ‘manager’ are found to not have any meaningful impact in classifying you in the higher salary bracket. It is however effective in shifting the average salary per class by a reasonable margin (around a few thousand £). This is likely because all jobs with ‘senior’ or ‘manager’ are likely already in the higher salary bracket pre modeling. 

**Step 4: Model evaluation**

Despite the models not all doing that well in terms of scores, there perhaps could be some use for the best performer, of which I’ve identified to be the CVEC gradient boosting model. It has:

- CV Mean Score: 0.7562850557049559
- Accuracy: 75.4673%
- Log Loss: 0.5519492435465978

These metrics can help alongside the confusion matrix & classification reports, which show us that the model is more likely to misclassify higher paying jobs as low paying, which can be beneficial as opposed to the other way around. So even with an imperfect accuracy, it’s a sly way to keep your customers/clients happy & improve customer retention by predicting that they’ll get a lower paying job but in actuality getting a higher paying one. Your marketing team could spin that to a selling point e.g. “We’re more conservative with your prospects compared to others”, or something like that.

Further inspection with the ROC-AUC & Precision Recall curves look to support this too, with roughly similar performance across both and in the train & test sets.
