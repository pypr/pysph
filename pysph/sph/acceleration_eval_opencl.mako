<%def name="do_group(helper, g_idx, sg_idx, group)" buffered="True">
#######################################################################
## Iterate over destinations in this group.
#######################################################################
% for dest, (eqs_with_no_source, sources, all_eqs) in group.data.items():
// Destination ${dest}
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
## Do any loop interactions between source and destination.
###################################################################
% if eq_group.has_loop():
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
#######################################################################
## Update NNPS locally if needed
#######################################################################
% if group.update_nnps:
<% helper.call_update_nnps(group) %>
% endif
% endfor
</%def>

#define abs fabs
#define max(x, y) fmax((double)(x), (double)(y))

__constant double pi=M_PI;
${helper.get_header()}

#######################################################################
## Iterate over groups
#######################################################################
% for g_idx, group in enumerate(helper.object.mega_groups):
% if len(group.data) > 0:
// ------------------------------------------------------------------
// Group${g_idx}
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
