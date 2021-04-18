# Stock Sentiment Analysis

#### Bryce Woods and Nicholas LaMonica

A stock sentiment analysis program that attempts
to predict the movements of stocks based on the prevailing sentiment from social media websites (twitter, reddit and stocktwits).

## Directions to Run

1. `pip install -r requirements.txt`
2. Change `sample_config.yaml` to `config.yaml` and fill in your api keys and other configurations.
3. Run `python main.py`

## Code explanation 
### Data
* The sentiment data will come from Twitter API. 
    * https://developer.twitter.com/en/docs
* The stock data will come from Yahoo Finance Python API
    * https://pypi.org/project/yfinance/
* Right now we are using tweet data from kaggle to train the ML model for now, and use the Twitter API later for further testing.

### Data Cleaning
We clean the tweets by removing the @ mentions, hashtags, ticker symbols, and other
unneeded symbols that don't have an effect on the sentiment of a tweet.   
Also need a way to ignore tweets from bots and only use users. 
Twitter API has a way to tell if a tweet was made from a phone app, web client, or from the API.

### Exploratory Data Analysis
None really needed so far. 

### ML Models
* We will be use Recurrent Neural Nets, specifically LSTM and GRU version of Recurrent Neural Nets. LSTM and GRU are optimized versions of regular RNNs.  
* We are using RNNs because they are good at adapting to sequence of data for predictions where the order of that data matters. 
* RNNs are  commonly used in time series data, this is perfect for our use case because we are analyzing the sentiment of the tweets overtime.  
* RNNs are also good for natural language processing because the order of the words that were written carries significance in predicting sentiment.
* We will validate our algorithm by testing it's performance against the movement of the stock.

### Results and Analysis
TBD

## Sources
* Papers: 
    * https://arxiv.org/pdf/1812.04199.pdf  
    * https://arxiv.org/pdf/1607.01958.pdf  
    * https://journalofbigdata.springeropen.com/articles/10.1186/s40537-017-0111-6
    * https://link.springer.com/article/10.1007/s42979-020-0076-y
* Tutorials/explanations: 
    * https://medium.com/@gabriel.mayers/sentiment-analysis-from-tweets-using-recurrent-neural-networks-ebf6c202b9d5
    * https://www.youtube.com/watch?v=LHXXI4-IEns
    * https://www.youtube.com/watch?v=8HyCNIVRbSU
    * https://www.youtube.com/watch?v=6niqTuYFZLQ
* API Docs: 
    * Twitter API: https://developer.twitter.com/en/docs
    * Yahoo finance python package: https://pypi.org/project/yfinance/
  