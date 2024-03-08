from meta.data_processor import DataProcessor
from agents.stablebaselines3_models import DRLAgent as DRLAgent_sb3
# from agents.stablebaselines3_models import DRLEnsembleAgent as ENSAgent_sb3
import os
def train(start_date, end_date, ticker_list, data_source, time_interval,
          technical_indicator_list, drl_lib, env, model_name, if_vix=True,
          **kwargs):
    
    


    #process data using unified data processor
    DP = DataProcessor(data_source, start_date, end_date, time_interval, **kwargs)
    price_array, tech_array, turbulence_array = DP.run(ticker_list,
                                                        technical_indicator_list,
                                                        if_vix, cache=True)
    data_config = {'price_array': price_array,
                   'tech_array': tech_array,
                   'turbulence_array': turbulence_array} 
    
    #build environment using processed data
    env_instance = env(config=data_config)
    env_instance.set_flag(True)


    #read parameters and load agents
    current_working_dir = kwargs.get('current_working_dir','/')
    # print(f'current_working_dir = {current_working_dir}')
    # exit()
    if drl_lib == 'stable_baselines3':
        total_timesteps = kwargs.get('total_timesteps', 1e6) #The total number of samples (env steps) to train on
        # print(f'total_timesteps = {total_timesteps}')
        agent_params = kwargs.get('agent_params')

        agent = DRLAgent_sb3(env = env_instance)

        model = agent.get_model(model_name, model_kwargs = agent_params)
        trained_model = agent.train_model(model=model,
                                tb_log_name=model_name,
                                total_timesteps=total_timesteps)
        print('Training finished!')
        trained_model.save(os.path.join(current_working_dir,model_name))
        env_instance.save_df_train()
        print('Trained model saved in ' + str(current_working_dir))
    else:
        raise ValueError('DRL library input is NOT supported. Please check.')