import jax
import jax.numpy as jnp


def sanity_check_models():
    import Models.Policy
    import Models.PolicyContinous
    import Models.QAdvantageous
    import Models.QModel
    import Models.Value
    import Models.model_utils
    
    x = jnp.ones((1, 4))
    pi = Models.Policy.Pi(4)
    pi_cont = Models.PolicyContinous.Pi_cont()
    q = Models.QModel.Q(4)
    q_adv = Models.QAdvantageous.Q_adv(4)
    v = Models.Value.V()
    
    Models.model_utils.show_summary(pi, [x])
    Models.model_utils.show_summary(pi_cont, [x])
    Models.model_utils.show_summary(q, [x])
    Models.model_utils.show_summary(q_adv, [x])
    Models.model_utils.show_summary(v, [x])
    
def sanity_check_buffers():
    import Buffers.UniformBuffer
    
    buffer = Buffers.UniformBuffer.UniformReplayBuffer(16, 4)
    state = jnp.ones((4,))
    action = 0
    reward = 0
    next_state = jnp.ones((4,))
    done = False
    
    for i in range(100):
        buffer.buffer_state = buffer.add(buffer.buffer_state, (state*i, action, reward, next_state*(100-i), done), i)
        print(buffer.buffer_state)
    
    key = jax.random.PRNGKey(0)
    for i in range(100):
        experiences, key = buffer.sample(key, buffer.buffer_state, 16)
        print(buffer.buffer_state)
        print(experiences)
            
    
    

def main():
    pass

if __name__ == "__main__":
    # sanity_check_models()
    sanity_check_buffers()