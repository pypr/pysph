<%def name="do_group(helper, g_idx, sg_idx, group)" buffered="True">
% for dest, (eqs_with_no_source, sources, all_eqs) in group.data.items():
// Destination ${dest}
% if all_eqs.has_initialize():
// Initialization for destination ${dest}
${helper.get_initialize_kernel(g_idx, sg_idx, group, dest, all_eqs)}
% endif

% if len(eqs_with_no_source.equations) > 0:
% if eqs_with_no_source.has_loop():
// Equations with no sources.
${helper.get_simple_loop_kernel(g_idx, sg_idx, group, dest, all_eqs)}
% endif
% endif

% for source, eq_group in sources.items():
// Source ${source}.
% if eq_group.has_loop():
${helper.get_loop_kernel(g_idx, sg_idx, group, dest, source, eq_group)}
% endif
% endfor

% if all_eqs.has_post_loop():
// post_loop for destination ${dest}
${helper.get_post_loop_kernel(g_idx, sg_idx, group, dest, all_eqs)}
% endif
// Finished destination ${dest}.
% if group.update_nnps:
<% helper.call_update_nnps(group) %>
% endif
% endfor
</%def>

#define abs fabs
#define pi M_PI
${helper.get_header()}

% for g_idx, group in enumerate(helper.object.mega_groups):
% if len(group.data) > 0:
// ------------------------------------------------------------------
// Group${g_idx}
% if group.has_subgroups:
% for sg_idx, sub_group in enumerate(group.data):
// Subgroup ${sg_idx}
${do_group(helper, g_idx, sg_idx, sub_group)}
% endfor # sg_idx
% else:
${do_group(helper, g_idx, -1, group)}
% endif # has_subgroups
% endif
// Finished Group${g_idx}
// ------------------------------------------------------------------
% endfor  # (for g_idx, group ...)
