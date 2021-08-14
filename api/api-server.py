import numpy as np
import os
from flask import Flask, request, render_template, make_response, url_for
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import unidecode
import random
import time
app = Flask(__name__, static_url_path='/static')

# Function that append a new applicant to the database
def novo_talento(csv, infos):
    df_talentos = pd.read_csv(csv, encoding='ISO-8859-1')
    aplicante_id = df_talentos['AplicanteID'].max()+1
    new_df = pd.DataFrame(infos, columns = ['Nome',	'Sobrenome','Email','Cidade','Estado','LadoAplicacao','TipoTrabalho','Tecnologias','MelhorTecnologia','Ingles','ExperienciaTrabalho','DescricaoExperiencia'])	

    new_df['AplicanteID'] = aplicante_id
    new_df = pd.concat([df_talentos, new_df])
    
    
    new_df.to_csv('aplicantes.csv', encoding='ISO-8859-1', index = False)
    return new_df

# Function that append a new job to the database
def nova_vaga(csv, infos):
    df_vagas = pd.read_csv(csv, encoding='ISO-8859-1')
    vaga_id = df_vagas['VagaID'].max()+1
    new_df = pd.DataFrame(infos, columns = ['NomeEmpresa','Setor','Cidade','Estado','NomeVaga','LadoAplicacao','TipoTrabalho','TecnologiasNecessarias','Ingles','InglesObrigatorio','Experiencia','DescricaoVaga'])
    new_df['VagaID'] = vaga_id
    new_df = pd.concat([df_vagas, new_df])
    
    new_df.to_csv('empresas.csv', encoding='ISO-8859-1', index = False)
    return new_df

# Funtion to clean data, remove accents and special characters
def limpa_dados(txt):
    # Remove acentos
    #print(txt)
    txt = unidecode.unidecode(str(txt))
    try:
        txt = txt.replace(",", " ")
        txt = txt.replace(".", " ")
        txt = txt.replace('*', " ")
    except:
        pass

    txt = txt.lower()
    
    return txt


# Function that recommends the best k_best applicants for a job
def recomendador(aplicantes, vagas, k_melhores = 5):
    aplicantes_df = aplicantes.copy()
    aplicantes_df['all_concat'] = aplicantes_df['all_concat'] = aplicantes_df['Cidade'] + " " + aplicantes_df['Estado'] + " " +  aplicantes_df['LadoAplicacao']+ " " + aplicantes_df['TipoTrabalho'] + " " +  aplicantes_df['Tecnologias'] + " "  + aplicantes_df['MelhorTecnologia'] + " " + aplicantes_df['Ingles']+ " " + aplicantes_df['ExperienciaTrabalho'] + " "  + aplicantes_df['DescricaoExperiencia'] 
    aplicantes_df['all_concat'] = aplicantes_df['all_concat'].apply(lambda x: limpa_dados(x))
    
    # Download toolkit
    nltk.download('rslp')
    nltk.download('stopwords')    

    stemmer = nltk.stem.RSLPStemmer()
    stop = nltk.corpus.stopwords.words('portuguese')

    # Remove stopwords in portuguese
    texto_aplicantes = aplicantes_df['all_concat'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    # Word tokenize
    texto_aplicantes = texto_aplicantes.apply(lambda x : x.split(" "))
    
    # Apply the stemmer
    texto_aplicantes = texto_aplicantes.apply(lambda x : [stemmer.stem(y) for y in x])
    texto_aplicantes = texto_aplicantes.apply(lambda x : " ".join(x))
    
    final_aplicantes = pd.DataFrame()
    final_aplicantes['text'] = texto_aplicantes
    final_aplicantes['AplicanteID'] = aplicantes_df['AplicanteID']
    
    # Apply TFIDF
    tfidf_vectorizer = TfidfVectorizer()

    tfidf_app = tfidf_vectorizer.fit_transform((final_aplicantes['text'])) #fitting and transforming the vector
    
    # ----------- End of applicants section -------------
    
    vagas_df = vagas.copy()
    # Pick up the last job registered
    vagas_df = vagas_df.tail(1)
    

    # Processing data
    vagas_df['all_concat'] = vagas_df['Cidade'] + " " + vagas_df['Estado'] + " " + vagas_df['Setor'] + " " + vagas_df['NomeVaga'] + " "  +  vagas_df['LadoAplicacao'] +  vagas_df['TipoTrabalho'] + " " + vagas_df['TecnologiasNecessarias'] + " " + vagas_df['Ingles'] + " " + vagas_df['Experiencia'] + " " + vagas_df['TipoTrabalho']
    vagas_df['all_concat'] = vagas_df['all_concat'].apply(lambda x: limpa_dados(x))
    
    #fitting and transforming the vector
    vaga_tfidf = tfidf_vectorizer.transform(vagas_df['all_concat']) 
    
    output = list(map(lambda x: cosine_similarity(vaga_tfidf, x),tfidf_app))
    distances = []
    for i in range(0,len(final_aplicantes)):
        distances.append(output[i][0][0])
    
    # Return the best applicants for the job    
    final_aplicantes['distances'] = distances
    top = final_aplicantes.sort_values('distances', ascending = False)['AplicanteID'][:k_melhores].values
    
    return top
    
# Flask api   
    
# Index html page    
@app.route('/')
def display_gui():
    return render_template('index.html')

# Formulary of applicants html page
@app.route('/formulario_talentos')
def display_form_talentos():
    return render_template('formulario_talento.html')

# Formulary jobs html page
@app.route('/formulario_empresas')
def display_form_empresas():
    return render_template('formulario_empresa.html')

# Page for showing the results
@app.route('/resultados')
def display_resultados():
    return render_template('resultado.html')

# Random recommendation
@app.route('/aleatorio')
def display_aleatoria():
    # Just to clean the html page every requistion	
    with open('templates/aleatorio.html', 'w') as f:
        f.write(" ")
    f.close()
    
    
    # Pick up a random ID (between 1 and 4)
    random_id = random.randint(1, 4)
    print('RANDOM INT', random_id)
    # Lê e recomenda
    df_aplicantes = pd.read_csv('aplicantes.csv', encoding='ISO-8859-1')
    df_vagas = pd.read_csv('empresas.csv', encoding='ISO-8859-1')
    df_vagas = df_vagas[df_vagas['VagaID'] == random_id]
 
 	# Return the best matches
    top_candidatos = recomendador(df_aplicantes, df_vagas)
    
    
    output_df = df_aplicantes.set_index('AplicanteID').loc[top_candidatos]

  

    # Page with table
    html_string = '''
                <!DOCTYPE html>
                <html lang="pt-br">
                <head>
                <meta charset="UTF-8">
                <meta http-equiv="X-UA-Compatible" content="IE=edge">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <link rel="stylesheet" type="text/css" href="../static/resultado_final3.css">
                <title>Recomendacao aleatoria - Triagem Inteligente para Recrutadores</title>
                </head>
                <body>
              
                
                <div>
                    <h1 style="text-align:left" id="titulo">Baseado na vaga aleatoria:</h1>

                </div>
                {table_1}

            
                <div>
                    <h1 style="text-align:left" id="titulo">Resultado da Triagem Inteligente</h1>
                    <p id="subtitulo">Com a nossa inteligencia artifical, voce pode selecionar os melhores talentos do mercado!</p>
                    <br>

                </div>
                {table_2}
                </body>
                </html>.
                '''
 
    
    
    print(df_vagas.tail(1))
    
    print('\n')
    
    print(output_df)
    
    time.sleep(0.5)
    
    
    # OUTPUT AN HTML FILE
    with open('templates/aleatorio_{}.html'.format(random_id), 'w') as f:
        f.write(html_string.format(table_1=df_vagas.tail(1).to_html(index=False), table_2=output_df.to_html(index=False)))
    f.close()
    
   

    print("\n")
    
    time.sleep(5)
    
    return render_template('aleatorio_{}.html'.format(random_id))

# Submit a new applicant
@app.route('/talentos', methods=['POST'])
def submeter_talento():
    nome = request.form['nome']
    sobrenome = request.form['sobrenome']
    email = request.form['email']
    cidade = request.form['cidade']
    estado = request.form['estado']
    lado_app = request.form['devweb']
    tipo_trab = request.form['vaga_talento']
    tecs = request.form.getlist('linguagens')
    tecs = "".join(tec + " " for tec in tecs)
    melhor_tec = request.form['melhor_linguagem']
    nivel_ingles = request.form['niv_ingles']
    senioridade = request.form.getlist('senioridade')
    senioridade = "".join(sen + " " for sen in senioridade)
    descricao = request.form['experiencia']
    
    user = np.array([[limpa_dados(nome), limpa_dados(sobrenome), email, limpa_dados(cidade), estado, lado_app, tipo_trab, tecs, melhor_tec, nivel_ingles, senioridade, limpa_dados(descricao)]])
    
    novo_talento('aplicantes.csv', user)
    print('Novo talento adicionado!')
    
    print("\n")


    return render_template('formulario_talento.html')


# Submit a new job
@app.route('/vagas', methods=['POST'])
def submeter_vaga():
    nome_empresa = request.form['nome']
    setor_empresa = request.form['setor']
    cidade = request.form['cidade']
    estado = request.form['estado']
    nome_vaga = request.form['nome_vaga']
    lado_app = request.form['devweb']
    tipo_trab = request.form['vaga_empresa']
    tecs = request.form.getlist('linguagens')
    tecs = "".join(tec + " " for tec in tecs)
    nivel_ingles = request.form['niv_ingles']
    ingles_obg = request.form['ingles_obg']
    senioridade = request.form.getlist('senioridade')
    senioridade = "".join(sen + " " for sen in senioridade)
    descricao = request.form['experiencia']
    

    vaga = np.array([[limpa_dados(nome_empresa), limpa_dados(setor_empresa), limpa_dados(cidade), estado, limpa_dados(nome_vaga), lado_app, tipo_trab, tecs, nivel_ingles, ingles_obg, senioridade, limpa_dados(descricao)]])
    
    nova_vaga('empresas.csv', vaga)  
    print('Nova vaga adicionada!')
    
    
    df_vaga_atual = pd.DataFrame(vaga, columns = ['NomeEmpresa','Setor','Cidade','Estado','NomeVaga','LadoAplicacao','TipoTrabalho','TecnologiasNecessarias','Ingles','InglesObrigatorio','Experiencia','DescricaoVaga'] )
    
    # Lê e recomenda
    df_aplicantes = pd.read_csv('aplicantes.csv', encoding='ISO-8859-1')

 
    top_candidatos = recomendador(df_aplicantes, df_vaga_atual)
    print(top_candidatos)
    
    output_df = df_aplicantes.set_index('AplicanteID').loc[top_candidatos]

    
    
    with open('templates/resultado.html', 'w') as f:
        f.write("")
    f.close()

    
    html_string = '''
                <!DOCTYPE html>
                <html lang="pt-br">
                <head>
                <meta charset="UTF-8">
                <meta http-equiv="X-UA-Compatible" content="IE=edge">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <link rel="stylesheet" type="text/css" href="../static/resultado_final3.css">
                <title>Cadastro de Talento - Triagem Inteligente para Recrutadores</title>
                </head>
                <body>
              
                
                <div>
                    <h1 style="text-align:left" id="titulo">Baseado na vaga que voce colocou:</h1>

                </div>
                {table_1}

            
                <div>
                    <h1 style="text-align:left" id="titulo">Resultado da Triagem Inteligente</h1>
                    <p id="subtitulo">Com a nossa inteligencia artifical, voce pode selecionar os melhores talentos do mercado!</p>
                    <br>

                </div>
                {table_2}
                </body>
                </html>.
                '''
                
    time.sleep(4)
    
    print(output_df)
    
    random_id = random.randint(1, 4)
    # OUTPUT AN HTML FILE
    with open('templates/resultado_{}.html'.format(random_id), 'w') as f:
        f.write(html_string.format(table_1=df_vaga_atual.tail(1).to_html(index=False), table_2=output_df.to_html(index=False)))
    f.close()
    print("\n")



    return render_template('resultado_{}.html'.format(random_id))





if __name__ == "__main__":
        port = int(os.environ.get('PORT', 80))
        app.run(host='0.0.0.0', port=port)