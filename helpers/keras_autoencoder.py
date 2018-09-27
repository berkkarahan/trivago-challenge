import numpy as np
from keras.layers import Input, Dense
from keras.models import Model, Sequential

class AutoEncoderReducer():

    def __init__(self, X, y, num_layers=13,
                 shape_reduction = 0.2
                 ):
        self.num_layers = num_layers
        self.shape_reduction = shape_reduction
        self.X = X
        self._size_list = []
        self._final_encoder = None
        self._final_decoder = None
        self.train_model = None
        self.encode_model = None

        #private bool parameters
        self._train_model_compiled = False

    #PRIVATE METHODS
    def _build_neuron_sizes(self):
        prev = self.X.shape[1]
        for _ in range(0,self.num_layers):
            itm = np.floor(prev - prev*self.shape_reduction)

            if np.abs(self._nextpower2(itm) - itm) > np.abs(self._prevpower2(itm) - itm):
                itm = self._prevpower2(itm)
            else:
                itm = self._nextpower2(itm)

            self._size_list.append(itm)
            prev = itm
        return self

    def _nextpower2(self,x):
        return 1 if x == 0 else 2**np.ceil(np.log2(x))

    def _prevpower2(self,x):
        return 1 if x == 0 else 2**np.floor(np.log2(x))

    #PUBLIC METHODS
    def build(self):
        self._build_neuron_sizes()

        #encoding model
        e_mdl = Sequential()
        #decoding model
        d_mdl = Sequential()

        in_sh = np.int(self.X.shape[1])
        #mdl = Sequential()
        e_mdl.add(Dense(in_sh, activation='linear', input_dim=in_sh))

        #add encoding layers
        for i in range(0,self.num_layers):
            sh_enc = np.int(self._size_list[i])
            e_mdl.add(Dense(sh_enc, activation='relu'))

        #final encoder
        fin_size = np.int(np.floor(self._size_list[len(self._size_list)-1]*(1-self.shape_reduction)))
        final_encoder = Dense(fin_size, activation = 'tanh')
        self._final_encoder = final_encoder
        e_mdl.add(final_encoder)
        self.encode_model = e_mdl

        #build above encoding model
        d_mdl.add(e_mdl)
        #add decoding layers
        for i in range(0,self.num_layers):
            sh_dec = np.int(self._size_list[self.num_layers-(i+1)])
            d_mdl.add(Dense(sh_dec, activation='relu'))

        #final decoding layer
        final_decoder = Dense(self.X.shape[1], activation = 'sigmoid')
        self._final_decoder = final_decoder
        d_mdl.add(final_decoder)

        #compile and save model
        d_mdl.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.train_model = d_mdl
        self._train_model_compiled = True
        return self

    def summary(self):
        if self._train_model_compiled:
            return self.train_model.summary()
        else:
            pass

    def fit(self, nb_epoch = 16, batch_size = 32):
        if self._train_model_compiled:
            self.train_model.fit(self.X,
                                 self.X,
                                 nb_epoch=nb_epoch,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 verbose=1)

    def transform(self, X):
        if self._train_model_compiled:
            return self.encode_model.predict(X)
