def _patched_put_accent(self, word):
        lower_word = word.lower()
        inputs = self.tokenizer(lower_word, return_tensors="np")
        inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
        
        # Add token_type_ids if missing (zeros with same shape as input_ids)
        if 'token_type_ids' not in inputs:
            inputs['token_type_ids'] = np.zeros_like(inputs['input_ids'])
        
        outputs = self.session.run(None, inputs)
        output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}
        logits = outputs[output_names["logits"]]
        probabilities = softmax(logits)
        scores = np.max(probabilities, axis=-1)[0]
        labels = np.argmax(logits, axis=-1)[0]
        pred_with_scores = [{'label': self.id2label[str(label)], 'score': float(score)} 
                            for label, score in zip(labels, scores)]

        stressed_word = self.render_stress(word, pred_with_scores)
        return stressed_word