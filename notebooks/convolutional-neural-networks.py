# Arquitetura de uma Rede Neural Convolucional

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import matplotlib.pyplot as plt

"""
    Dado que o range de valores possível para um pixel vai de 0-255 escalonamos o valor entre 0-1.
    Esse processo tornará nosso modelo menos variante a pequenas alterações.
"""
x_train = x_train / 255
x_test = x_test / 255

model = Sequential()
model.add(Conv2D(32, (5,5), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (5,5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

"""
    Usada na camada de saída do classificador, na qual, realmente estamos tentando gerar as probabilidades
    para definir a classe de cada entrada.
"""

optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())


"""
    Reduz o parâmetro de 'Learning Rate' caso não haja melhorias em um número específico de épocas.
    Muito útil para encontrar o mínimo global.	
"""

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

batch_size = 32
epochs = 10

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[learning_rate_reduction])

history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
range_epochs = range(1, len(acc) + 1)

plt.style.use('default')
accuracy_val = plt.plot(range_epochs, val_acc, label='Acurácia de Validação')
accuracy_train = plt.plot(range_epochs, acc, label='Acurácia de Treino', linestyle='--')
plt.setp(accuracy_val, linewidth=2, marker='o', markersize=5)
plt.setp(accuracy_train, linewidth=2, marker='o', markersize=5)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()