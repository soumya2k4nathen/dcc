from flask import Flask, render_template, request
import analysis

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = {}
    if request.method == 'POST':
        analysis_type = request.form['analysis']
        if analysis_type == 'univariate':
            mean_std, mode = analysis.univariate_analysis()
            result = {'type': 'univariate', 'mean_std': mean_std, 'mode': mode}
        elif analysis_type == 'bivariate':
            plots, chi = analysis.bivariate_analysis()
            result = {'type': 'bivariate', 'plots': plots, 'chi': chi}
        elif analysis_type == 'multivariate':
            plots, svm_r, nb_r, mlr = analysis.multivariate_analysis()
            result = {'type': 'multivariate', 'plots': plots, 'svm': svm_r, 'nb': nb_r, 'mlr': mlr}
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
