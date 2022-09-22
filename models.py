import keras.backend
from bert4keras.models import *
from config import config


class GAU_alpha(RoFormerV2):
    """GAU-α
    改动：基本模块换成GAU
    链接：https://kexue.fm/archives/9052
    """
    def initializer(self, shape, dtype=None, order=3, gain=1.0):
        return super(GAU_alpha, self).initializer(shape, dtype, order, gain)

    def get_inputs(self):
        """重写get_inputs，输入仅为token_ids"""
        x_in = self.apply(
            layer=Input, shape=(self.sequence_length, 80), name='Encoder-Input-Feature'
        )
        inputs = [x_in]

        return inputs

    def apply_embeddings(self, inputs):
        inputs = inputs[:]  # 浅拷贝
        x = inputs.pop(0)
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=x,
            layer=Masking,
            name="Encoder-Masking"
        )
        # subsampling
        x = self.apply(
            inputs=x,
            layer=Lambda,
            function=lambda x: K.expand_dims(x, -1),
            name='Encoder-Expand-Dim'
        )

        x = self.apply(
            inputs=x,
            layer=Conv2D,
            filters=144,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            kernel_initializer=self.initializer,
            name='Encoder-Subsampling-1'
        )
        x = self.apply(
            inputs=x,
            layer=Conv2D,
            filters=144,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            kernel_initializer=self.initializer,
            name='Encoder-Subsampling-2'
        )
        x = self.apply(
            inputs=x,
            layer=Reshape,
            target_shape=(-1, K.int_shape(x)[-2] * K.int_shape(x)[-1]),
            name='Encoder-Reshape'
        )
        x = self.apply(
            inputs=x,
            layer=Dense,
            units=self.hidden_size,
            kernel_initializer=self.initializer,
            name='Encoder-Dense'
        )

        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Embedding-Norm'
        )

        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )

        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return x

    def apply_main_layers(self, inputs, index):
        """GAU-α 的主体是基于Gated Attention Unit的模块
        顺序：GAU  --> Add --> LN
        """
        x = inputs

        attention_name = 'Transformer-%d-GatedAttentionUnit' % index
        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(x)

        # Self Attention
        xi = x
        x = [x, position_bias]
        arguments = {'a_bias': None, 'p_bias': 'rotary'}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.insert(1, attention_mask)
        x = self.apply(
            inputs=x,
            layer=GatedAttentionUnit,
            arguments=arguments,
            units=self.intermediate_size,
            key_size=self.attention_key_size,
            activation=self.hidden_act,
            use_bias=True,
            normalization='squared_relu',
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            name='%s-Norm' % attention_name
        )

        return x

    def apply_final_layers(self, inputs):
        """剩余部分
        """
        x = inputs

        x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=2048,
                    use_bias=True,
                    kernel_initializer=self.initializer,
                    activation='relu',
                    name='Projection'
                )

        # x = self.apply(
        #             inputs=x,
        #             layer=Dense,
        #             units=256,
        #             use_bias=True,
        #             kernel_initializer=self.initializer,
        #             activation='softmax',
        #             name='CTC-Layer'
        #         )

        return x


class GAU_alphaV2(GAU_alpha):
    def apply_main_layers(self, inputs, index):
        """GAU-α 的主体是基于Gated Attention Unit的模块
        顺序：GAU  --> Add --> LN
        """
        x = inputs

        attention_name = 'Transformer-%d-GatedAttentionUnit' % index
        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(x)
        if index > self.num_hidden_layers // 2:
            pl = 1.0 - index / self.num_hidden_layers * 0.3
        else:
            pl = 1.0

        # Self Attention
        xi = x
        x = [x, position_bias]
        arguments = {'a_bias': None, 'p_bias': 'rotary'}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.insert(1, attention_mask)
        x = self.apply(
            inputs=x,
            layer=GatedAttentionUnit,
            arguments=arguments,
            units=self.intermediate_size,
            key_size=self.attention_key_size,
            activation=self.hidden_act,
            use_bias=True,
            normalization='squared_relu',
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Lambda,
            function=self.stochastic_survival,
            arguments={'p_survival': pl},
            name='Stochastic-Depth'
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            name='%s-Norm' % attention_name
        )

        return x

    def stochastic_survival(self, y, p_survival=1.0):
        # binomial random variable
        survival = K.random_binomial((1,), p=p_survival)
        # during testing phase:
        # - scale y (see eq. (6))
        # - p_survival effectively becomes 1 for all layers (no layer dropout)
        return K.in_test_phase(y,
                               1.0 / p_survival * survival * y)

if __name__ == '__main__':
    GAU = GAU_alpha(
        vocab_size=None,
        hidden_size=config.hidden_size, # 512
        num_hidden_layers=config.n_layer, # 16
        num_attention_heads=config.n_head, # 1
        attention_key_size=config.attention_key_size,
        intermediate_size=config.inter_hidden_size, # 1024
        hidden_act=config.hidden_act, # 'swish'
        dropout_rate=config.dropout_rate, # 0.1
        attention_dropout_rate=config.attention_dropout_rate, # 0.1
        max_position=512
    )
    GAU.build()
    GAU.model.summary()
    inp = np.ones((2, 16, 80))
    print(GAU.model.predict(inp))