import tensorflow as tf
from tensorflow.python.ops.losses import losses_impl

class DocunetLoss(tf.keras.losses.CategoricalCrossentropy):
    """Implementation of the DocUnet loss functions as presented in http://openaccess.thecvf.com/content_cvpr_2018/papers/Ma_DocUNet_Document_Image_CVPR_2018_paper.pdf"""

    def __init__(self, lamda=0.1, reduction=losses_impl.ReductionV2.NONE):
        """
        Initialize the loss function

        Args:
            lamda (float): coefficient of second loss term
            reduction (object): type of reduction, set to none as we perform it manually
        """

        super(DocunetLoss, self).__init__(reduction=reduction)
        self.lamda = lamda

    def call(self, target, output):
        """
        Computes the loss between the target and provided output

        Args:
            target (tf.Tensor): y_true, the target tensor
            output (tf.Tensor): y_pred, the predicted tensor

        Returns:
            tf.Tensor: loss
        """

        x = target[:, :, :, 0]
        y = target[:, :, :, 1]
        back_sign_x, back_sign_y = tf.cast(tf.math.equal(x, -1 * tf.ones(x.numpy().shape)), tf.float32), tf.cast(tf.math.equal(y, -1 * tf.ones(y.numpy().shape)), tf.float32)
        # assert back_sign_x == back_sign_y

        back_sign = tf.cast(tf.math.equal(tf.math.add(back_sign_x, back_sign_y), 2 * tf.ones(x.numpy().shape)), tf.float32)
        fore_sign = tf.math.add(-1 * back_sign, tf.ones(back_sign.numpy().shape, tf.float32))

        loss_term_1_x = tf.math.reduce_sum(tf.math.abs(tf.math.subtract(output[:, :, :, 0], x)) * fore_sign) / tf.math.reduce_sum(fore_sign)
        loss_term_1_y = tf.math.reduce_sum(tf.math.abs(tf.math.subtract(output[:, :, :, 1], y)) * fore_sign) / tf.math.reduce_sum(fore_sign)
        loss_term_1 = tf.math.add(loss_term_1_x, loss_term_1_y)

        loss_term_2_x = tf.math.abs(tf.math.reduce_sum(tf.math.subtract(output[:, :, :, 0], x) * fore_sign)) / tf.math.reduce_sum(fore_sign)
        loss_term_2_y = tf.math.abs(tf.math.reduce_sum(tf.math.subtract(output[:, :, :, 1], y) * fore_sign)) / tf.math.reduce_sum(fore_sign)
        loss_term_2 = tf.math.add(loss_term_2_x, loss_term_2_y)

        zeros_x = tf.zeros(x.numpy().shape)
        zeros_y = tf.zeros(y.numpy().shape)

        loss_term_3_x = tf.math.maximum(zeros_x, output[:, :, :, 0])
        loss_term_3_y = tf.math.maximum(zeros_y, output[:, :, :, 1])
        loss_term_3 = tf.math.reduce_sum(tf.math.add(loss_term_3_x, loss_term_3_y) * back_sign) / tf.math.reduce_sum(back_sign)

        loss = loss_term_1 - self.lamda * loss_term_2 + loss_term_3
        loss = losses_impl._safe_mean(loss, target.numpy().shape[0])

        return loss