from flask import Flask, render_template, send_from_directory
import os
from analysis import perform_univariate, perform_bivariate, perform_multivariate

app = Flask(__name__)

# Path to the static folder to store the images
app.config['UPLOAD_FOLDER'] = 'static/'

@app.route('/')
def index():
    # Perform all analyses
    univariate_result, univariate_plots = perform_univariate()
    bivariate_result, bivariate_plots = perform_bivariate('feature1', 'feature2')
    multivariate_result, multivariate_plots = perform_multivariate()

    # Combine results and plot paths
    all_results = univariate_result + bivariate_result + multivariate_result
    all_plots = univariate_plots + bivariate_plots + multivariate_plots

    # Return results to the template
    return render_template('index.html', results=all_results, plots=all_plots)

@app.route('/static/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
