from __future__ import annotations
import minedojo


'''

'''


if __name__=="__main__": 

    env = minedojo.make(
        task_id="harvest_wool_with_shears_and_sheep",
        image_size=(160, 256)
    )
    obs = env.reset()