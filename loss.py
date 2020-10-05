from torch import nn
class LossFn():
    def __init__(self):
        self.intent_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.req_slot_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.status_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.cat_value_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.non_cat_value_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def get_intent_loss(self, y_pred, y_true):
        loss = self.intent_loss_fn(y_pred, y_true)
        return loss

    def get_req_slot_loss(self, y_pred, y_true, mask):
        loss = self.req_slot_loss_fn(y_pred, y_true) * mask
        loss = loss.sum() / mask.sum()
        return loss

    def get_status_loss(self, y_pred, y_true, mask):
        loss = self.status_loss_fn(y_pred, y_true).sum()
        return loss

    def get_cat_value_loss(self,y_pred, y_true, mask):
        loss = self.cat_value_loss_fn(y_pred, y_true) * mask
        loss = loss.sum()
        if mask.sum() > 1.:
            loss = loss / mask.sum()
        return loss

    def get_noncat_value_loss(self,y_pred, y_true):
        loss = self.non_cat_value_loss_fn(y_pred, y_true)
        return loss