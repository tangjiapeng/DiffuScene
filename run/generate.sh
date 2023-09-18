cd ./scripts

python  generate_diffusion.py ../config/diffusion/diffusion_bedrooms_dim512_nomask_instancond_objfeats_lat32.yaml \
 /Your experiment directory/diffusion_bedrooms_dim512_nomask_instancond_objfeats_lat32/gen_top2down_notexture_nofloor \
 /Your dataset directory/3d_front_processed/bedrooms/threed_future_model_bedroom.pkl   \
    --weight_file /Your experiment directory/diffusion_bedrooms_dim512_nomask_instancond_objfeats_lat32/model_38000  \
    --without_screen  --n_sequences 1000 --render_top2down --save_mesh --no_texture --without_floor