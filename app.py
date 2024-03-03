from flask import Flask, render_template, url_for, request, redirect
from flask import send_file
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import io
import base64
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import plotly.graph_objs as go

app = Flask(__name__)

json_data = ''
@app.route('/', methods=['POST', 'GET'])
def index():
    userdata = convert_pandas_json()
    # userdata = json_data
    # print(userdata[1])
    print("#####")
    # print(userdata[1]['customer_id'])
    givenId = 1
    oneUser = ''
    for sample in userdata:
        # print(sample['id'])
        if givenId == sample['customer_id']:
            oneUser = sample
    return render_template('home.html', userData = oneUser, recentTransactions = getRecentTransactions(1, userdata))

def convert_pandas_json():
    with open('datasets/data.json', 'r') as file:
            # Load the JSON data
        json_data = json.load(file)
    return json_data
        
    # print(json_data)
    
def getRecentTransactions(givenId, userdata):
    print("***")
    # print(userdata)
    allTransactions = []
    recentTransactions = []
    for transaction in userdata:
        print(transaction)
        if transaction.get('customer_id') == givenId:
            allTransactions.append(transaction)
            if len(recentTransactions) < 5:
                recentTransactions.append(transaction)
            else:
                # Sort recentTransactions by date in descending order
                recentTransactions.sort(key=lambda x: x.get('date'), reverse=True)
                # If the current transaction is more recent than the oldest one in recentTransactions, replace it
                if transaction.get('date') > recentTransactions[-1].get('date'):
                    recentTransactions[-1] = transaction
    
    print("Recent Transactions:")
    for transaction in recentTransactions:
        print(transaction)
                    
    return recentTransactions

def execute_str_python_code(code):
    code += "\nplt.savefig('images/plot.png', format='png')"
    exec(code)

def fetch_code_from_string(response):
    start_index = response.find("```python") + len("```python")
    end_index = response.rfind("```")
    python_code = response[start_index:end_index].strip()
    print('YUVI')
    print(python_code)
    return python_code

def draw_line(user, expense):
    plt.plot(expense)
    plt.savefig('images/line.png', format='png')
    plt.xlabel('AaronMurray')
    plt.ylabel('Amount')

@app.route('/debit/<int:id>', methods=['POST', 'GET'])
def debit(id):
    if request.method=='POST':
        df = pd.read_json('datasets/data.json')
        df_user = df[df['customer_id']==id]
        json_user_list = df_user.to_dict(orient='records')
        transactions = [jsn['amt'] for jsn in json_user_list]
        dates = [ jsn['date'].date().isoformat() for jsn in json_user_list]
        json_user_list = json_user_list[0]
        json_user_list['dates'] = dates
        json_user_list['transactions'] = transactions
        draw_line('AaronMurray', df[df['Name']=='AaronMurray']['amt'])
        sorted_lists = sorted(zip(dates, transactions), key=lambda x: x[0])
        dates, transactions = zip(*sorted_lists)
        print(dates)
        print(transactions)

        return render_template('debit.html', json_dates = dates, json_transactions=transactions)
    else:
        return render_template('debit.html')
    
@app.route('/recommend', methods=['POST', 'GET'])
def recommend():
    output = call_gpt()
    ajout = 'sdsdfdfsbzdfb'
    category = "essentials"  # or "healthcare" based on your choice
    ratingsData = userRatingsData(category)
    company_ratings = {}

    # Aggregate ratings for each company
    for product in ratingsData['products']:
        for company, rating in product['prices'].items():
            if company not in company_ratings:
                company_ratings[company] = []
            company_ratings[company].append(rating)

    # Calculate average rating for each company
    average_ratings = {}
    for company, ratings in company_ratings.items():
        average_ratings[company] = sum(ratings) / len(ratings)

    # Determine which company has the best rating
    best_company = max(average_ratings, key=average_ratings.get)

    sorted_average_ratings = dict(sorted(average_ratings.items(), key=lambda item: item[1], reverse=True))
    return render_template('recommend.html', output = output, average_ratings=sorted_average_ratings)


def call_gpt():
    user = 'AaronMurray'
    text = 'Each line should end with \n. From the given data give me best suggestion in 3 lines with short content to minimize expenses for'+user
    # text2 = "from above"
    with open('datasets/data.json', 'r') as file:
        json_data = json.load(file)
    # print(json_data)
    # print(text+str(json_data[:50])+text2)

    openai = OpenAI(
        api_key="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"
    )

    completion = openai.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {
                "role": "user",
                "content": text+str(json_data[:20]),
            },
        ],
    )
    output = completion.choices[0].message.content
    with open("output.txt", "w") as file:
        file.write(output)
    return output
    # code = fetch_code_from_string(output)
    # execute_str_python_code(code)

def userRatingsData(category):
    with open('datasets/ratingsData.json', 'r') as file:
        # Load the JSON data
        data = json.load(file)
    return data['categories'][category]

@app.route('/recommendations', methods=['POST', 'GET'])
def suggestTheCompanyInCategory():
    

    return render_template('recommendations.html', best_company=best_company, average_ratings=sorted_average_ratings)



nltk.download('vader_lexicon')
#  portfolio analysis,
# market trend monitoring, and
#personalized news recommendations tailored to individual investors'portfolio
 
sectors = ['Financial Services', 'Technology and Innovation', 'Healthcare', 'Manufacturing', 'Energy', 'Agriculture', 'Retail and Consumer Goods', 'Education', 'Entertainment and Media', 'Transportation']
companies_data = [
    {"name": "Qualcomm Inc", "key_product": "Semiconductors, Wireless Technology"},
    {"name": "Tyson Foods", "key_product": "Meat and Poultry Products"},
    {"name": "AstraZeneca PLC", "key_product": "Pharmaceutical Drugs"},
    {"name": "Google", "key_product": "Search Engine"}
]
sid = SentimentIntensityAnalyzer()
 
 
 
@app.route('/invest')
def index_invest():
    return render_template('index_invest.html')
 
 
@app.route('/analyze',)
def analyze_portfolio():
  return render_template('Portfolio.html')
 
@app.route('/openai')
def openai():
    key_products = [(company['name'], company['key_product']) for company in companies_data]
    return render_template('OpenAi.html', key_products=key_products)
 
openai.api_key = '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'
 
@app.route('/chatgpt', methods=['POST','GET'])
def chatgpt():
   if request.method == 'POST' or request.method == 'GET':
        company = request.args.get('company')
        product = request.args.get('product')
 
        if company:
            text = f"Generate the latest news articles related to {company}"
        elif product:
            text = f"Generate the latest news articles related to {product}"
        else:
            text = "Generate the latest news articles related to semiconductors"
        # text = "Generate the latest news articles related to semicondutor"
        openai = OpenAI(
            api_key="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"
        )
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )
        
        output_raw = completion.choices[0].message.content
        output = output_raw.split("\n")
        score=analyze_nltk(output)
        
        print(output)
        print(score)
        return render_template('OpenAi.html',output=output,score=score)
   else:
        return render_template('OpenAi.html',output="",score="")
 
 
def analyze_nltk(output):
    sid = SentimentIntensityAnalyzer()
    
    # Join the list of strings into a single string
    text = ' '.join(output)
    
    # Analyze sentiment using VADER
    sentiment_scores = sid.polarity_scores(text)
    
    # Return the sentiment analysis result
    return sentiment_scores
 
@app.route('/market')
def market():
    # Data for the pie chart
    data = {
        'labels': ['Financial Services', 'Technology and Innovation', 'Healthcare', 'Manufacturing', 'Energy', 'Agriculture', 'Retail and Consumer Goods','Education','Entertainment and Media','Transportation'],
        'values': [8, 7, 18, 12, 7, 2, 11, 8, 3, 9] , # Example data (percentage distribution)
        'size_layout': 'responsive',  # Size layout option
         
    }
 
    if request.method == 'POST' or request.method == 'GET':
        sectors = [
            'Financial Services',
            'Technology and Innovation',
            'Healthcare',
            'Manufacturing',
            'Energy',
            'Agriculture',
            'Retail and Consumer Goods',
            'Education',
            'Entertainment and Media',
            'Transportation'
        ]
 
 
        openai = OpenAI(
            api_key="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"
        )
        growth_data = {}
        for sector in sectors:
            completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": f"Generate the latest growth information for {sector}.",
                },
            ],
        )
            output_raw = completion.choices[0].message.content
            growth_data[sector] = output_raw
            print(growth_data)
 
        
    return render_template('result.html', data=data,sectors=sectors, growth_data=growth_data)
 
# @app.route('/detgraph')
# def detgraph():
#     # Load data from CSV
#     df = pd.read_csv('static/det.csv')
#     df=df.head(10)
#     print(df)
    
#     # Convert 'Record Date' column to datetime format
#     df['Record Date'] = pd.to_datetime(df['Record Date'])
    
#     # Create line chart
#     plt.figure(figsize=(10, 6))
#     for column in df.columns[1:]:
#         plt.plot(df['Record Date'], df[column], label=column)
#     plt.xlabel('Record Date')
#     plt.ylabel('Debt')
#     plt.title('Debt Over Time')
#     plt.legend()
#     plt.grid(True)
    
#     # Save the plot to a temporary file
#     plot_path = 'static/line_chart.png'
#     plt.savefig(plot_path)
    
#     # Close the plot to free memory
#     plt.close()
    
#     # Pass the plot path to the template
#     return render_template('result.html', plot_path=plot_path)
 
@app.route('/show_chart')
def show_chart():
    return send_file('line_chart.html')
 
 
 
@app.route('/growth', methods=['POST','GET'])
def get_sector_growth():
 
    if request.method == 'POST' or request.method == 'GET':
        sectors = [
            'Financial Services',
            'Technology and Innovation',
            'Healthcare',
            'Manufacturing',
            'Energy',
            'Agriculture',
            'Retail and Consumer Goods',
            'Education',
            'Entertainment and Media',
            'Transportation'
        ]
 
 
        openai = OpenAI(
            api_key="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"
        )
        growth_data = {}
        for sector in sectors:
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": f"Generate the latest growth information for {sector}",
                    },
                ],
            )
            output_raw = completion.choices[0].message.content
            growth_data[sector] = output_raw
            print(growth_data)
        
        return render_template('result.html', sectors=sectors, growth_data=growth_data)
    else:
        return render_template('result.html', sectors=[], growth_data={})
 
    
 

if __name__ == "__main__":
    app.run(debug=True)




# key = "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"