import unittest
import os
import sys

datasets_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(datasets_dir)

from datasets.umich_contact_dataset import UmichContactDataset
import torch
import numpy as np

class TestUmichContactDataset(unittest.TestCase):
    """
    Used to test methods in the UmichContactDataset.
    """

    def test_leg_f1_score_calculation(self):
        """
        Test that the Leg F1-Score calculation is correct, and matches up
        with what is expected.
        """

        # Initialize the dataset class
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = UmichContactDataset(data_name="test.npy",
                                        label_name="test_label.npy", train_ratio=0.85,
                                        augment=False, use_class_imbalance_w=False,
                                        window_size=150, device=device,
                                        partition="training_splitted")
        
        # Define the input y_pred and y_gt
        y_pred = torch.tensor(
        [[7.18903254e-10, 9.13594135e-06, 7.33426063e-10, 9.32049961e-06,
        4.77328554e-45, 6.06597015e-41, 4.86971231e-45, 6.18851088e-41,
        3.89475642e-05, 4.94952082e-01, 3.97343572e-05, 5.04950778e-01,
        2.58599254e-40, 3.28632206e-36, 2.63823305e-40, 3.35271017e-36,],
        [0.00000000e+00, 3.43962697e-01, 0.00000000e+00, 1.47044198e-01,
        0.00000000e+00, 3.43962697e-01, 0.00000000e+00, 1.47044198e-01,
        0.00000000e+00, 6.29989654e-03, 0.00000000e+00, 2.69320844e-03,
        0.00000000e+00, 6.29989654e-03, 0.00000000e+00, 2.69320844e-03,],
        [2.49794350e-06, 1.83935391e-23, 1.00582507e-10, 7.40636555e-28,
        1.65855385e-41, 1.22127162e-58, 6.67835376e-46, 4.91758765e-63,
        9.99957238e-01, 7.36315795e-18, 4.02644037e-05, 2.96485842e-22,
        6.63939327e-36, 4.88889920e-53, 2.67342643e-40, 1.96857029e-57,],
        [0.00000000e+00, 6.87925393e-01, 0.00000000e+00, 2.94088397e-01,
        0.00000000e+00, 1.69918285e-79, 0.00000000e+00, 7.26401388e-80,
        0.00000000e+00, 1.25997931e-02, 0.00000000e+00, 5.38641688e-03,
        0.00000000e+00, 3.11216195e-81, 0.00000000e+00, 1.33045055e-81,],
        [5.88590171e-10, 7.47990117e-06, 6.00480482e-10, 7.63100520e-06,
        3.90804874e-45, 4.96641293e-41, 3.98699656e-45, 5.06674112e-41,
        3.89476945e-05, 4.94953738e-01, 3.97344901e-05, 5.04952467e-01,
        2.58600119e-40, 3.28633305e-36, 2.63824188e-40, 3.35272138e-36,],
        [0.00000000e+00, 6.23806791e-13, 0.00000000e+00, 1.93070537e-12,
        0.00000000e+00, 6.23806791e-13, 0.00000000e+00, 1.93070537e-12,
        0.00000000e+00, 1.22099006e-01, 0.00000000e+00, 3.77900994e-01,
        0.00000000e+00, 1.22099006e-01, 0.00000000e+00, 3.77900994e-01,],
        [2.04514409e-06, 1.50593630e-23, 8.23500287e-11, 6.06382199e-28,
        1.35790966e-41, 9.99893089e-59, 5.46777608e-46, 4.02618206e-63,
        9.99957690e-01, 7.36316129e-18, 4.02644219e-05, 2.96485977e-22,
        6.63939628e-36, 4.88890141e-53, 2.67342764e-40, 1.96857118e-57,],
        [0.00000000e+00, 1.24761358e-12, 0.00000000e+00, 3.86141073e-12,
        0.00000000e+00, 3.08161848e-91, 0.00000000e+00, 9.53772453e-91,
        0.00000000e+00, 2.44198012e-01, 0.00000000e+00, 7.55801988e-01,
        0.00000000e+00, 6.03171621e-80, 0.00000000e+00, 1.86683874e-79,]], dtype=torch.float64, device=device)
        y_gt = torch.tensor([15, 13, 6, 4, 3, 14, 8, 9], device=device)

        # Run through the f1-score method
        f1score_of_legs = dataset.calculate_f1_score_of_legs(y_pred, y_gt)

        # Calculate the expected (using a different method for verification)
        predictions = np.argmax(y_pred.cpu().numpy(), axis=1).astype(np.int64)
        pred_b = np.array([list(np.binary_repr(x).rjust(4, '0')) for x in predictions], dtype=np.int64)
        gt_b = np.array([list(np.binary_repr(x).rjust(4, '0')) for x in y_gt.cpu().numpy()], dtype=np.int64)

        f1score_of_legs_des = []
        for i in range(0, 4):
            p = pred_b[:,i]
            g = gt_b[:,i]

            TP = np.sum([1 for p, g in zip(p, g) if p == 1 and g == 1])
            FP = np.sum([1 for p, g in zip(p, g) if p == 1 and g == 0])
            TN = np.sum([1 for p, g in zip(p, g) if p == 0 and g == 0])
            FN = np.sum([1 for p, g in zip(p, g) if p == 0 and g == 1])

            if TP + FP == 0: precision = 0
            else: precision = TP / (TP + FP)

            if TP + FN == 0: recall = 0
            else: recall = TP / (TP + FN)

            if precision + recall == 0: f1_score = 0
            else: f1_score = 2 * precision * recall / (precision + recall)
            f1score_of_legs_des.append(f1_score)

        # Compare with the expected
        np.testing.assert_array_equal(f1score_of_legs, f1score_of_legs_des)

if __name__ == "__main__":
    unittest.main()