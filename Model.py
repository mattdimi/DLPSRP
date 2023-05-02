class Model:

    def __init__(self, 
                name,
                type,
                model_class,
                is_rnn = False,
                ):
        """Utility class to define a model

        Args:
            name (str): Model name
            type (str): `classification` | `regression`
            model_class (_type_): Model instance. Needs to support `fit` and `predict` methods.
        """
        self.name = name

        assert(type in ["classification", "regression"])
        self.type = type
        
        assert(all([hasattr(model_class, "fit"), hasattr(model_class, "predict")]))
        self.cls = model_class # to test for hyperparameters, we can use a list of Model instances with different hyperparameters

        self.is_rnn = is_rnn