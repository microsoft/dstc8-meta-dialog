from mldc.data.data_handler import RAW_TEXT


class BeamsToTextFormatting:
  def format_prediction(self, predictions, scores, context, target_meta):
    te = self.data_handler.text_embedder
    for bidx, dlg in enumerate(predictions):
      res = []
      for beam in dlg:
        res.append(dict(prediction=te.decode_ids_as_tokens(beam['tokens'].tolist()),
                        words=[te.decode_ids_as_tokens(turn.tolist()) for turn in context[RAW_TEXT][bidx]],
                        score=beam['score']))
      yield res
