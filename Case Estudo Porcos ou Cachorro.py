from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

##Meu modelo
porco1 = [0,1,0]
porco2 = [0,1,1]
porco3 = [1,1,0]

cachorro1 = [0,1,1]
cachorro2 = [1,0,1]
cachorro3=  [1,1,1]

dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]

# 1 = cachorro 2 = porco

classes = [ 1,1,1,0,0,0]

#instanciando o LinearSVC
model = LinearSVC()
#Criando um modelo utilizando o FIT
model.fit(dados, classes)

#Criando um animal com atributos alguns atributos e pedindo pro o modelo predizer se ele é cachorro = 1 ou porco = 0
animal_misterioso = [1,1,1]
model.predict([animal_misterioso])

misterio1 = [1,1,1]
misterio2 = [1,1,0]
misterio3 = [0,1,1]

testes = [misterio1, misterio2, misterio3]
previsoes = model.predict(testes)

testes_classes = [0,1,1]

#Calculando minha acuracia na mão
corretos = (previsoes == testes_classes).sum()
total = len(testes)
taxa_de_acerto = corretos/total
print(f'Taxa de acerto : {taxa_de_acerto * 100}')


#Calculando minha acuracia utilizando a função accuracy_score do skleanr
taxa_de_acerto = accuracy_score(testes_classes, previsoes)
print(f'Taxa de acerto : {taxa_de_acerto * 100}')

