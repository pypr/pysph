<%def name="do_group(helper, g_idx, sg_idx, group)" buffered="True">
#######################################################################
## Call any `pre` functions
#######################################################################
% if group.pre:
<% helper.call_pre(group) %>
% endif
#######################################################################
## Iterate over destinations in this group.
#######################################################################
% for dest, (eqs_with_no_source, sources, all_eqs) in group.data.items():
// Destination ${dest}
## Call py_initialize if it is defined for the equations.
<% helper.call_py_initialize(all_eqs, dest) %>
#######################################################################
## Initialize all equations for this destination.
#######################################################################
% if all_eqs.has_initialize():
// Initialization for destination ${dest}
${helper.get_initialize_kernel(g_idx, sg_idx, group, dest, all_eqs)}
% endif
#######################################################################
## Handle all the equations that do not have a source.
#######################################################################
% if len(eqs_with_no_source.equations) > 0:
% if eqs_with_no_source.has_loop():
// Equations with no sources.
${helper.get_simple_loop_kernel(g_idx, sg_idx, group, dest, eqs_with_no_source)}
% endif
% endif
#######################################################################
## Iterate over sources.
#######################################################################
% for source, eq_group in sources.items():
// Source ${source}.
###################################################################
## Do any pairwise initializations.
###################################################################
% if eq_group.has_initialize_pair():
// Initialization for destination ${dest} with source ${source}.
${helper.get_initialize_pair_kernel(g_idx, sg_idx, group, dest, source, eq_group)}
% endif
###################################################################
## Do any loop interactions between source and destination.
###################################################################
% if eq_group.has_loop() or eq_group.has_loop_all():
${helper.get_loop_kernel(g_idx, sg_idx, group, dest, source, eq_group)}
% endif
% endfor
###################################################################
## Do any post_loop assignments for the destination.
###################################################################
% if all_eqs.has_post_loop():
// post_loop for destination ${dest}
${helper.get_post_loop_kernel(g_idx, sg_idx, group, dest, all_eqs)}
% endif
###################################################################
## Do any reductions for the destination.
###################################################################
% if all_eqs.has_reduce():
<% helper.call_reduce(all_eqs, dest) %>
% endif
// Finished destination ${dest}.
% endfor
#######################################################################
## Update NNPS locally if needed
#######################################################################
% if group.update_nnps:
<% helper.call_update_nnps(group) %>
% endif
#######################################################################
## Call any `post` functions
#######################################################################
% if group.post:
<% helper.call_post(group) %>
% endif
<% helper.end_group(group) %>
</%def>

#define abs fabs
#define max(x, y) fmax((double)(x), (double)(y))
#define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

${helper.get_header()}
#######################################################################
## Iterate over groups
#######################################################################
% for g_idx, group in enumerate(helper.object.mega_groups):
% if len(group.data) > 0:
// ------------------------------------------------------------------
// Group${g_idx}
% if group.condition is not None:
<% helper.check_condition(group) %>
% endif
#######################################################################
## Start iteration if needed.
#######################################################################
% if group.iterate:
<% helper.start_iteration(group) %>
% endif
#######################################################################
## Handle sub-groups.
#######################################################################
% if group.has_subgroups:
% for sg_idx, sub_group in enumerate(group.data):
// Subgroup ${sg_idx}
% if sub_group.condition is not None:
<% helper.check_condition(sub_group) %>
% endif
${do_group(helper, g_idx, sg_idx, sub_group)}
% endfor ## sg_idx
% else:
${do_group(helper, g_idx, -1, group)}
% endif ## has_subgroups
#######################################################################
## Stop iteration if needed.
#######################################################################
% if group.iterate:
<% helper.stop_iteration(group) %>
% endif
##
% endif ## len(group.data) > 0:
// Finished Group${g_idx}
// ------------------------------------------------------------------
% endfor  ## (for g_idx, group ...)
