from typing import Any, Dict, List, Optional
import logging
import numpy as np
import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.modules import Highway
from allennlp.nn.activations import Activation
from allennlp.modules.feedforward import FeedForward
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import masked_softmax
from allennlp.training.metrics.drop_em_and_f1 import DropEmAndF1
from torch.autograd import Variable
logger = logging.getLogger(__name__)


@Model.register("RAIN")
class Bert(Model):
    """
    This class augments the QANet model with some rudimentary numerical reasoning abilities, as
    published in the original DROP paper.
    The main idea here is that instead of just predicting a passage span after doing all of the
    QANet modeling stuff, we add several different "answer abilities": predicting a span from the
    question, predicting a count, or predicting an arithmetic expression.  Near the end of the
    QANet model, we have a variable that predicts what kind of answer type we need, and each branch
    has separate modeling logic to predict that answer type.  We then marginalize over all possible
    ways of getting to the right answer through each of these answer types.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 dropout_prob: float = 0.1,
                 ffn_hidden_size: int =768,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 answering_abilities: List[str] = None) -> None:
        super().__init__(vocab, regularizer)


        #print (vocab)

        if answering_abilities is None:
            self.answering_abilities = ["span_extraction",
                                        "addition_subtraction", "counting"]
        else:
            self.answering_abilities = answering_abilities



        
        
        text_embed_dim = text_field_embedder.get_output_dim()

        self.W = torch.nn.Linear(text_embed_dim*2,text_embed_dim)
        
        self._text_field_embedder = text_field_embedder

        #self._embedding_proj_layer = torch.nn.Linear(text_embed_dim, encoding_in_dim)

        """
            为了用于self attention
        """

        if len(self.answering_abilities) > 1:
            self._answer_ability_predictor = FeedForward(text_embed_dim,
                                                         activations=[Activation.by_name('relu')(inplace=True),
                                                                      Activation.by_name('linear')()],
                                                         hidden_dims=[ffn_hidden_size,
                                                                      len(self.answering_abilities)],
                                                         num_layers=2,
                                                         dropout=dropout_prob)

        if "span_extraction" in self.answering_abilities:
            self._span_extraction_index = self.answering_abilities.index("span_extraction")
            self._span_start_predictor = FeedForward(text_embed_dim,
                                                      activations=[Activation.by_name('relu')(inplace=True),
                                                                   Activation.by_name('linear')()],
                                                      hidden_dims=[ffn_hidden_size,1],
                                                      num_layers=2,
                                                      dropout=dropout_prob)
            self._span_end_predictor = FeedForward(text_embed_dim ,
                                                      activations=[Activation.by_name('relu')(inplace=True),
                                                                   Activation.by_name('linear')()],
                                                      hidden_dims=[ffn_hidden_size,1],
                                                      num_layers=2,
                                                      dropout=dropout_prob)


        if "addition_subtraction" in self.answering_abilities:
            self._addition_subtraction_index = self.answering_abilities.index("addition_subtraction")
            self._number_sign_predictor = FeedForward(text_embed_dim*2,
                                                      activations=[Activation.by_name('relu')(inplace=True),
                                                                   Activation.by_name('linear')()],
                                                      hidden_dims=[ffn_hidden_size,3],
                                                      num_layers=2,
                                                      dropout=dropout_prob)

        if "counting" in self.answering_abilities:
            self._counting_index = self.answering_abilities.index("counting")
            self._count_number_predictor = FeedForward(text_embed_dim,
                                                       activations=[Activation.by_name('relu')(inplace=True),
                                                                    Activation.by_name('linear')()],
                                                       hidden_dims=[ffn_hidden_size, 10],
                                                       num_layers=2,
                                                       dropout=dropout_prob)
        



        self._drop_metrics = DropEmAndF1()
        self._dropout = torch.nn.Dropout(p=dropout_prob)

        initializer(self)

    def forward(self,  # type: ignore
                passage_question: Dict[str, torch.LongTensor],
                number_indices: torch.LongTensor,
                answer_type = None,
                answer_as_spans: torch.LongTensor = None,
                answer_as_add_sub_expressions: torch.LongTensor = None,
                answer_as_counts: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:


        passage_question_mask = passage_question["mask"].float()
        embedded_passage_question = self._dropout(self._text_field_embedder(passage_question))# Encode with bert

        batch_size = embedded_passage_question.size(0)

        encoded_passage_question = embedded_passage_question
        """
        passage_vactor 用 [CLS]对应的代替
        """

        passage_question_vector = encoded_passage_question[:,0] 

        if len(self.answering_abilities) > 1:
            # Shape: (batch_size, number_of_abilities)
            answer_ability_logits = \
                self._answer_ability_predictor(passage_question_vector)
            answer_ability_log_probs = torch.nn.functional.log_softmax(answer_ability_logits, -1)
            #best_answer_ability = torch.argmax(answer_ability_log_probs, 1)

        if "counting" in self.answering_abilities:
            # Shape: (batch_size, 10)
            count_number_logits = self._count_number_predictor(passage_question_vector)
            count_number_log_probs = torch.nn.functional.log_softmax(count_number_logits, -1)
            # Info about the best count number prediction
            # Shape: (batch_size,)
            best_count_number = torch.argmax(count_number_log_probs, -1)
            best_count_log_prob = \
                torch.gather(count_number_log_probs, 1, best_count_number.unsqueeze(-1)).squeeze(-1)
            if len(self.answering_abilities) > 1:
                best_count_log_prob += answer_ability_log_probs[:, self._counting_index]

        if "span_extraction" in self.answering_abilities:
            # Shape: (batch_size, passage_length)
            span_start_logits = self._span_start_predictor(encoded_passage_question).squeeze(-1)
            # Shape: (batch_size, passage_length)
            span_end_logits = self._span_end_predictor(encoded_passage_question).squeeze(-1)
            # Shape: (batch_size, passage_length)
            span_start_log_probs = util.masked_log_softmax(span_start_logits, passage_question_mask)
            span_end_log_probs = util.masked_log_softmax(span_end_logits, passage_question_mask)

            # Info about the best passage span prediction
            span_start_logits = util.replace_masked_values(span_start_logits, passage_question_mask, -1e7)
            span_end_logits = util.replace_masked_values(span_end_logits, passage_question_mask, -1e7)
            # Shape: (batch_size, 2)
            best_span = get_best_span(span_start_logits, span_end_logits)
            # Shape: (batch_size, 2)
            best_start_log_probs = \
                torch.gather(span_start_log_probs, 1, best_span[:, 0].unsqueeze(-1)).squeeze(-1)
            best_end_log_probs = \
                torch.gather(span_end_log_probs, 1, best_span[:, 1].unsqueeze(-1)).squeeze(-1)
            # Shape: (batch_size,)
            best_span_log_prob = best_start_log_probs + best_end_log_probs
            if len(self.answering_abilities) > 1:
                best_span_log_prob += answer_ability_log_probs[:, self._span_extraction_index]


        if "addition_subtraction" in self.answering_abilities:
            # Shape: (batch_size, # of numbers in the passage)
            number_indices = number_indices.squeeze(-1)
            number_mask = (number_indices != -1).long()


            clamped_number_indices = util.replace_masked_values(number_indices, number_mask, 0)
            #encoded_passage_for_numbers = torch.cat([modeled_passage_list[0], modeled_passage_list[3]], dim=-1)
            # Shape: (batch_size, # of numbers in the passage, encoding_dim)
            encoded_numbers = torch.gather(
                    encoded_passage_question,
                    1,
                    clamped_number_indices.unsqueeze(-1).expand(-1, -1, encoded_passage_question.size(-1)))
           

            #self._external_number_embedding = self._external_number_embedding.cuda(device)

            #encoded_numbers = self.self_attention(encoded_numbers,number_mask)
            encoded_numbers = self.Concat_attention(encoded_numbers,passage_question_vector,number_mask)
            # Shape: (batch_size, # of numbers in the passage)
            #encoded_numbers = torch.cat(
            #        [encoded_numbers, passage_question_vector.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1)], -1)

            # Shape: (batch_size, # of numbers in the passage, 3)
            number_sign_logits = self._number_sign_predictor(encoded_numbers)
            number_sign_log_probs = torch.nn.functional.log_softmax(number_sign_logits, -1)

            # Shape: (batch_size, # of numbers in passage).
            best_signs_for_numbers = torch.argmax(number_sign_log_probs, -1)
            # For padding numbers, the best sign masked as 0 (not included).
            best_signs_for_numbers = util.replace_masked_values(best_signs_for_numbers, number_mask, 0)
            # Shape: (batch_size, # of numbers in passage)
            best_signs_log_probs = torch.gather(
                    number_sign_log_probs, 2, best_signs_for_numbers.unsqueeze(-1)).squeeze(-1)
            # the probs of the masked positions should be 1 so that it will not affect the joint probability
            # TODO: this is not quite right, since if there are many numbers in the passage,
            # TODO: the joint probability would be very small.
            best_signs_log_probs = util.replace_masked_values(best_signs_log_probs, number_mask, 0)
            # Shape: (batch_size,)

        
            if len(self.answering_abilities) > 1:
                # batch_size
                best_combination_log_prob = best_signs_log_probs.sum(-1)
                best_combination_log_prob += answer_ability_log_probs[:, self._addition_subtraction_index]

            
        best_answer_ability = torch.argmax(torch.stack([best_span_log_prob,best_combination_log_prob,best_count_log_prob],-1),1) 


        output_dict = {}

        # If answer is given, compute the loss.
        if answer_as_spans is not None or answer_as_add_sub_expressions is not None or answer_as_counts is not None:

            log_marginal_likelihood_list = []

            for answering_ability in self.answering_abilities:
                if answering_ability == "span_extraction":
                    # Shape: (batch_size, # of answer spans)
                    gold_span_starts = answer_as_spans[:, :, 0]
                    gold_span_ends = answer_as_spans[:, :, 1]
                    # Some spans are padded with index -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    gold_span_mask = (gold_span_starts != -1).long()
                    clamped_gold_span_starts = \
                        util.replace_masked_values(gold_span_starts, gold_span_mask, 0)
                    clamped_gold_span_ends = \
                        util.replace_masked_values(gold_span_ends, gold_span_mask, 0)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_span_starts = \
                        torch.gather(span_start_log_probs, 1, clamped_gold_span_starts)
                    log_likelihood_for_span_ends = \
                        torch.gather(span_end_log_probs, 1, clamped_gold_span_ends)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_spans = \
                        log_likelihood_for_span_starts + log_likelihood_for_span_ends
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_spans = \
                        util.replace_masked_values(log_likelihood_for_spans, gold_span_mask, -1e7)
                    # Shape: (batch_size, )
#                    log_marginal_likelihood_for_span = torch.sum(log_likelihood_for_spans,-1) 
                    log_marginal_likelihood_for_span = util.logsumexp(log_likelihood_for_spans)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_span)

                elif answering_ability == "addition_subtraction":
                    # The padded add-sub combinations use 0 as the signs for all numbers, and we mask them here.
                    # Shape: (batch_size, # of combinations)
                    gold_add_sub_mask = (answer_as_add_sub_expressions.sum(-1) > 0).float()
                    # Shape: (batch_size, # of numbers in the passage, # of combinations)
                    gold_add_sub_signs = answer_as_add_sub_expressions.transpose(1, 2)
                    # Shape: (batch_size, # of numbers in the passage, # of combinations)
                    log_likelihood_for_number_signs = torch.gather(number_sign_log_probs, 2, gold_add_sub_signs)
                    # the log likelihood of the masked positions should be 0
                    # so that it will not affect the joint probability
                    log_likelihood_for_number_signs = \
                        util.replace_masked_values(log_likelihood_for_number_signs, number_mask.unsqueeze(-1), 0)
                    # Shape: (batch_size, # of combinations)
                    log_likelihood_for_add_subs = log_likelihood_for_number_signs.sum(1)
                    # For those padded combinations, we set their log probabilities to be very small negative value
                    log_likelihood_for_add_subs = \
                        util.replace_masked_values(log_likelihood_for_add_subs, gold_add_sub_mask, -1e7)
                    # Shape: (batch_size, )

                    #log_marginal_likelihood_for_add_sub =  torch.sum(log_likelihood_for_add_subs,-1)
                    #log_marginal_likelihood_for_add_sub = util.logsumexp(log_likelihood_for_add_subs)
                    #log_marginal_likelihood_list.append(log_marginal_likelihood_for_add_sub)
                    


                    
                    log_marginal_likelihood_for_add_sub = util.logsumexp(log_likelihood_for_add_subs)


                    #log_marginal_likelihood_for_external = util.logsumexp(log_likelihood_for_externals)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_add_sub)

                elif answering_ability == "counting":
                    # Count answers are padded with label -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    # Shape: (batch_size, # of count answers)
                    gold_count_mask = (answer_as_counts != -1).long()
                    # Shape: (batch_size, # of count answers)
                    clamped_gold_counts = util.replace_masked_values(answer_as_counts, gold_count_mask, 0)
                    log_likelihood_for_counts = torch.gather(count_number_log_probs, 1, clamped_gold_counts)
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_counts = \
                        util.replace_masked_values(log_likelihood_for_counts, gold_count_mask, -1e7)
                    # Shape: (batch_size, )
                    #log_marginal_likelihood_for_count =  torch.sum(log_likelihood_for_counts,-1)
                    log_marginal_likelihood_for_count = util.logsumexp(log_likelihood_for_counts)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_count)

                else:
                    raise ValueError(f"Unsupported answering ability: {answering_ability}")

            if len(self.answering_abilities) > 1:
                # Add the ability probabilities if there are more than one abilities
                all_log_marginal_likelihoods = torch.stack(log_marginal_likelihood_list, dim=-1)
                loss_for_type = -(torch.sum(answer_ability_log_probs*answer_type,-1)).mean()
                loss_for_answer = -(torch.sum(all_log_marginal_likelihoods,-1)).mean()
                loss = loss_for_type+loss_for_answer
            else:
                marginal_log_likelihood = log_marginal_likelihood_list[0]
                loss =  - marginal_log_likelihood.mean()
            output_dict["loss"] = loss

        # Compute the metrics and add the tokenized input to the output.
        if metadata is not None:
            output_dict["question_id"] = []
            output_dict["answer"] = []
            passage_question_tokens = []
            for i in range(batch_size):
                passage_question_tokens.append(metadata[i]['passage_question_tokens'])

                if len(self.answering_abilities) > 1:
                    predicted_ability_str = self.answering_abilities[best_answer_ability[i].detach().cpu().numpy()]
                else:
                    predicted_ability_str = self.answering_abilities[0]

                answer_json: Dict[str, Any] = {}

                # We did not consider multi-mention answers here
                if predicted_ability_str == "span_extraction":
                    answer_json["answer_type"] = "span"
                    passage_question_token = metadata[i]['passage_question_tokens']
                    #offsets = metadata[i]['passage_token_offsets']
                    predicted_span = tuple(best_span[i].detach().cpu().numpy())
                    start_offset = predicted_span[0]
                    end_offset = predicted_span[1]
                    predicted_answer = " ".join([token for token in passage_question_token[start_offset:end_offset+1] if token!="[SEP]"]).replace(" ##","")
                    answer_json["value"] = predicted_answer
                    answer_json["spans"] = [(start_offset, end_offset)]
                elif predicted_ability_str == "counting":
                    answer_json["answer_type"] = "count"
                    predicted_count = best_count_number[i].detach().cpu().numpy()
                    predicted_answer = str(predicted_count)
                    answer_json["count"] = predicted_count
                elif predicted_ability_str == "addition_subtraction":
                    answer_json["answer_type"] = "arithmetic"
                    original_numbers = metadata[i]['original_numbers']
                    sign_remap = {0: 0, 1: 1, 2: -1}
                    predicted_signs = [sign_remap[it] for it in best_signs_for_numbers[i].detach().cpu().numpy()]
                    result=0
                    for j,number in enumerate(original_numbers):
                        sign = predicted_signs[j]
                        if sign!=0:
                            result += sign * number
                    
                    predicted_answer = str(result)
                    #offsets = metadata[i]['passage_token_offsets']
                    number_indices = metadata[i]['number_indices']
                    #number_positions = [offsets[index] for index in number_indices]
                    answer_json['numbers'] = []
                    for indice, value, sign in zip(number_indices, original_numbers, predicted_signs):
                        answer_json['numbers'].append({'span': indice, 'value': str(value), 'sign': sign})
                    if number_indices[-1] == -1:
                        # There is a dummy 0 number at position -1 added in some cases; we are
                        # removing that here.
                        answer_json["numbers"].pop()
                    answer_json["value"] = str(result)
                else:
                    raise ValueError(f"Unsupported answer ability: {predicted_ability_str}")

                output_dict["question_id"].append(metadata[i]["question_id"])
                output_dict["answer"].append(answer_json)
                answer_annotations = metadata[i].get('answer_annotations', [])
                if answer_annotations:
                    self._drop_metrics(predicted_answer, answer_annotations)
            # This is used for the demo.
            #output_dict["passage_question_attention"] = passage_question_attention
            output_dict["passage_question_tokens"] = passage_question_tokens
            #output_dict["passage_tokens"] = passage_tokens
        return output_dict
    
    def self_attention(self,numbers,mask):
        # batch_size,1,#numbers,dim
        M = torch.unsqueeze(numbers,1)
        M = self.W(M)

        #batch_size,#numbers,#numbers,dim
        M = M * torch.unsqueeze(numbers,2)

        S = torch.sum(M,-1) 
        a = masked_softmax(S,mask)
        numbers_new = numbers + a.bmm(numbers)
        return numbers_new

    def Difference_self_attention(self,numbers,mask):
        
        """
        numbers: batch_size,#numbers,dim
        passage_quesstion_vector : batch_size, dim 
        """
        # batch_size,#numbers,#numbers,dim
        M = torch.unsqueeze(numbers,1) - torch.unsqueeze(numbers,2)
        #M = torch.unsqueeze(numbers,2) - torch.unsqueeze(numbers,1)
        #M = self._attention_weight(M)
        # batch_size,#numbers,#numbers,1
        M = self.W(M)
        # batch_size,#numbers,#numbers
        S = torch.squeeze(M,-1)
        #M = M * torch.unsqueeze(numbers,2)
        # batch_size,#numbers,#numbers
        #S = torch.sum(M,-1) 
        # batch_size,#numbers,#numbers
        a = masked_softmax(S,mask)
        
        #numbers_new = a.bmm(numbers)
        #numbers_new = numbers + a.bmm(numbers)
        numbers_new = torch.cat([numbers,a.bmm(numbers)],-1)

        return numbers_new
    
    def Concat_attention(self,numbers,passage_quesstion_vector,mask):
        
        """
        numbers: batch_size,#numbers,dim
        passage_quesstion_vector : batch_size, dim 
        """
        # batch_size,#numbers,#numbers,dim
        #M = torch.unsqueeze(numbers,1) - torch.unsqueeze(numbers,2)
        number_num = numbers.size(1)
        

        M = torch.cat([torch.unsqueeze(numbers,1).repeat(1,number_num,1,1), torch.unsqueeze(numbers,2).repeat(1,1,number_num,1)],-1)
        #M = self._attention_weight(M)
        M = self.W(M)
        M = M * torch.unsqueeze(torch.unsqueeze(passage_quesstion_vector,1),1)
        # batch_size,#numbers,#numbers
        S = torch.sum(M,-1) 
        # batch_size,#numbers,#numbers
        a = masked_softmax(S,mask)
        
        numbers_new = torch.cat([numbers,a.bmm(numbers)],-1)

        return numbers_new


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._drop_metrics.get_metric(reset)
        return {'em': exact_match, 'f1': f1_score}
