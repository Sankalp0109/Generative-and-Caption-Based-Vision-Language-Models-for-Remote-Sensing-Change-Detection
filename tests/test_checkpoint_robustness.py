import unittest
import tempfile
import pathlib
import torch
import torch.nn as nn
from src.training import save_checkpoint, load_checkpoint, get_checkpoint_epoch
from src.dataset import Vocabulary

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

class CheckpointRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_dir = pathlib.Path(self.temp_dir.name)
        self.model = SimpleModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.vocab = Vocabulary()
        self.device = torch.device("cpu")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_load_checkpoint_successfully(self):
        # Save a valid checkpoint
        checkpoint_path = save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=5,
            loss=0.45,
            vocab=self.vocab,
            checkpoint_dir=self.checkpoint_dir,
            name="test_model"
        )
        
        self.assertTrue(checkpoint_path.exists())
        
        # Verify epoch
        epoch = get_checkpoint_epoch(checkpoint_path, self.device)
        self.assertEqual(epoch, 5)
        
        # Load the checkpoint
        new_model = SimpleModel()
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
        
        loaded_model, loaded_optimizer, loaded_epoch, loaded_vocab, loaded_loss = load_checkpoint(
            model=new_model,
            optimizer=new_optimizer,
            checkpoint_path=checkpoint_path,
            device=self.device
        )
        
        self.assertEqual(loaded_epoch, 5)
        self.assertAlmostEqual(loaded_loss, 0.45)
        self.assertIsNotNone(loaded_vocab)

    def test_load_corrupted_checkpoint_renames_file_and_raises_error(self):
        checkpoint_path = self.checkpoint_dir / "corrupted_checkpoint.pt"
        
        # Write corrupted/invalid zip bytes to checkpoint_path
        with open(checkpoint_path, "wb") as f:
            f.write(b"not a valid zip file content")
            
        self.assertTrue(checkpoint_path.exists())
        
        # Verify that loading raises RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                checkpoint_path=checkpoint_path,
                device=self.device
            )
            
        self.assertIn("is corrupted", str(ctx.exception))
        
        # Verify that the corrupted checkpoint file was renamed to .pt.corrupted
        self.assertFalse(checkpoint_path.exists())
        corrupted_backup_path = checkpoint_path.with_suffix(".pt.corrupted")
        self.assertTrue(corrupted_backup_path.exists())

    def test_get_checkpoint_epoch_corrupted_checkpoint_renames_file_and_raises_error(self):
        checkpoint_path = self.checkpoint_dir / "corrupted_checkpoint_epoch.pt"
        
        with open(checkpoint_path, "wb") as f:
            f.write(b"")  # Empty file (corrupted)
            
        self.assertTrue(checkpoint_path.exists())
        
        with self.assertRaises(RuntimeError):
            get_checkpoint_epoch(checkpoint_path, self.device)
            
        self.assertFalse(checkpoint_path.exists())
        corrupted_backup_path = checkpoint_path.with_suffix(".pt.corrupted")
        self.assertTrue(corrupted_backup_path.exists())

if __name__ == "__main__":
    unittest.main()
