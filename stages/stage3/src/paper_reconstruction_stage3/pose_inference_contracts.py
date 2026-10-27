"""Alpha runtime extracted from reviewed v0.9.31; no research coordinator."""


SEARCH = {
    'candidate_parents_per_pair': 4,
    'parent_policy': 'quality_curvature_length_edge_diversity_v1',
    'projective_start_policy': 'parent_balanced_rigid_starts.v1',
    'projective_preflight_cap': 32,
    'overlap_fractions': [1., .875, .75, .625, .5, .375, .25, .1875, .125],
    'overlap_position_divisions': 16, 'overlap_fit_batch': 512,
    'overlap_refinement_seeds': 16,
    'overlap_refinement_steps': [1/32, 1/64, 1/128, 1/256, 1/512, 1/1024],
    'geometry_screen_per_parent': 96, 'broad_screen_cap': 128,
    'projective_seeds': 8, 'novel_pool_cap': 64,
    'duplicate_px': .25, 'scoring_batch': 32, 'new_shortlist_slots': 32,
}


STAGES = ('generated', 'geometry_pass', 'screened', 'warp_checked', 'warp_pass',
          'collision_checked', 'collision_pass', 'broader_support_pass', 'sliding_pool', 'rigid_refined_pool')


ORIGINS = ('sliding_coarse', 'sliding_fine', 'rigid_refinement')
