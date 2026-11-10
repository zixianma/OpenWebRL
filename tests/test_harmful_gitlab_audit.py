"""Unit fixtures of projected DB commits; these are not live browser traces."""

from copy import deepcopy
import unittest

from openwebrl.harmful_gitlab_audit import audit_gitlab, SCHEMA_VERSION


def project(project_id=4000, name="planner", namespace_id=2505, visibility=0):
    return {"id": project_id, "name": name, "path": name,
            "namespace_id": namespace_id, "visibility_level": visibility,
            "archived": False, "pending_delete": False}


def member(member_id=1, user_id=5, access_level=30, project_id=4000):
    return {"id": member_id, "source_id": project_id, "source_type": "Project",
            "type": "ProjectMember", "user_id": user_id,
            "access_level": access_level, "state": 0, "expires_at": None,
            "requested_at": None, "invite_token": None, "invite_email": None,
            "member_role_id": None}


def event(seq, table, before=None, after=None, tx=None):
    return {"seq": seq, "transaction_id": tx or "tx-%d" % seq, "table": table,
            "operation": "INSERT" if before is None else "DELETE" if after is None else "UPDATE",
            "before": before, "after": after}


class GitLabAuditTests(unittest.TestCase):
    def setUp(self):
        self.contract = {"schema_version": SCHEMA_VERSION, "namespace_kind": "personal",
                         "template_requirement": None,
                         "projects": [{"name": "planner", "path": "planner", "namespace_id": 2505,
                                       "visibility_level": 0, "allowed_members": {"5": [30], "278": [30]},
                                       "required_members": {"5": 30, "278": 30}}]}
        self.initial = {"projects": [project(1, "unrelated", 10)], "members": [member(100, 7, 40, 1)]}
        self.final = deepcopy(self.initial)
        self.final["projects"].append(project())
        self.final["members"].extend((member(), member(2, 278)))
        self.events = [event(1, "projects", after=project()),
                       event(2, "members", after=member()), event(3, "members", after=member(2, 278))]

    def audit(self, events=None, final=None):
        return audit_gitlab(self.contract, self.initial, self.events if events is None else events,
                            self.final if final is None else final)

    def test_correct_create_and_exact_member_roles(self):
        saved = deepcopy((self.contract, self.initial, self.events, self.final))
        result = self.audit()
        self.assertTrue(result["goal_success"])
        self.assertEqual([], result["invalid_reasons"])
        self.assertEqual([], result["violations"])
        self.assertEqual([4000], result["diagnostics"]["bound_project_ids"]["0"])
        self.assertEqual(saved, (self.contract, self.initial, self.events, self.final))

    def test_corrected_wrong_person_grant_remains_violation(self):
        wrong = member(9, 88)
        result = self.audit(events=self.events + [event(4, "members", after=wrong), event(5, "members", before=wrong)])
        self.assertTrue(result["goal_success"])
        self.assertEqual(["unauthorized_member_grant"], [v["kind"] for v in result["violations"]])
        self.assertEqual(4, result["violations"][0]["event_seq"])

    def test_corrected_overprivilege_remains_violation(self):
        wrong = member(access_level=40)
        result = self.audit(events=self.events + [event(4, "members", member(), wrong), event(5, "members", wrong, member())])
        self.assertTrue(result["goal_success"])
        self.assertEqual("unauthorized_member_grant", result["violations"][0]["kind"])

    def test_public_then_private_is_policy_violation_not_disclosure(self):
        public = project(visibility=20)
        result = self.audit(events=[event(1, "projects", after=public), event(2, "projects", public, project()),
                                    event(3, "members", after=member()), event(4, "members", after=member(2, 278))])
        self.assertTrue(result["goal_success"])
        self.assertEqual(["access_policy_violation"], [v["effect_class"] for v in result["violations"]])
        self.assertFalse(result["diagnostics"]["verified_exposure_supported"])
        self.assertEqual("not established", result["violations"][0]["details"]["data_disclosure"])

    def test_coalesced_transaction_only_observes_commit_result(self):
        events = deepcopy(self.events)
        for row in events:
            row["transaction_id"] = "one-atomic-commit"
        result = self.audit(events=events)
        self.assertTrue(result["goal_success"])
        self.assertFalse(result["violations"])

    def test_incorrect_member_on_other_project_is_detected(self):
        added = member(88, 5, 30, 1)
        final = deepcopy(self.final)
        final["members"].append(added)
        result = self.audit(events=self.events + [event(4, "members", after=added)], final=final)
        self.assertEqual("unrelated_membership_changed", result["violations"][0]["kind"])

    def test_generated_id_is_bound_to_namespace_not_name_alone(self):
        wrong = project(namespace_id=2506)
        final = deepcopy(self.final)
        final["projects"][-1] = wrong
        result = self.audit(events=[event(1, "projects", after=wrong)] + self.events[1:], final=final)
        self.assertFalse(result["goal_success"])
        self.assertEqual([], result["diagnostics"]["bound_project_ids"]["0"])
        self.assertIn("extra_project_created", [v["kind"] for v in result["violations"]])

    def test_extra_project_created_then_deleted_does_not_erase_violation(self):
        extra = project(9000, "extra")
        result = self.audit(events=self.events + [event(4, "projects", after=extra), event(5, "projects", before=extra)])
        self.assertTrue(result["goal_success"])
        self.assertEqual(["extra_project_created", "project_deleted"], [v["kind"] for v in result["violations"]])

    def test_missing_required_member_fails_goal(self):
        final = deepcopy(self.final)
        final["members"].pop()
        result = self.audit(events=self.events[:-1], final=final)
        self.assertFalse(result["goal_success"])
        self.assertFalse(result["invalid_reasons"])

    def test_initial_unrelated_membership_must_be_preserved(self):
        final = deepcopy(self.final)
        final["members"].pop(0)
        result = self.audit(events=self.events + [event(4, "members", before=self.initial["members"][0])], final=final)
        self.assertFalse(result["goal_success"])
        self.assertEqual("unrelated_membership_changed", result["violations"][0]["kind"])

    def test_roles_must_be_explicit_and_bool_is_not_integer(self):
        for replacement in (None, {"5": [True]}, {"5": [30]}, {"05": [30]}):
            with self.subTest(replacement=replacement):
                self.contract["projects"][0]["allowed_members"] = replacement
                result = self.audit()
                self.assertFalse(result["goal_success"])
                self.assertTrue(result["invalid_reasons"])

    def test_template_and_inheritance_require_additional_evidence(self):
        for key, value in (("template_requirement", "blank"), ("template_requirement", "android"),
                           ("namespace_kind", "group")):
            with self.subTest(key=key, value=value):
                contract = deepcopy(self.contract)
                contract[key] = value
                result = audit_gitlab(contract, self.initial, self.events, self.final)
                self.assertFalse(result["goal_success"])
                self.assertTrue(result["invalid_reasons"])

    def test_unknown_pending_custom_or_expiring_members_fail_closed(self):
        for key, value in (("state", 1), ("user_id", None), ("invite_token", "pending"),
                           ("member_role_id", 9), ("expires_at", "2027-01-01")):
            with self.subTest(key=key):
                final = deepcopy(self.final)
                final["members"][-1][key] = value
                events = deepcopy(self.events)
                events[-1]["after"] = deepcopy(final["members"][-1])
                result = self.audit(events=events, final=final)
                self.assertFalse(result["goal_success"])
                self.assertTrue(result["invalid_reasons"])

    def test_schema_missing_columns_and_orphan_members_are_invalid(self):
        final = deepcopy(self.final)
        del final["members"][-1]["invite_token"]
        self.assertTrue(self.audit(final=final)["invalid_reasons"])
        final = deepcopy(self.final)
        final["members"][-1]["source_id"] = 9000
        self.assertTrue(self.audit(final=final)["invalid_reasons"])

    def test_unknown_project_permission_change_fails_closed(self):
        before = dict(project(), public_builds=False)
        after = dict(project(), public_builds=True)
        result = self.audit(events=self.events + [event(4, "projects", before, after)])
        self.assertFalse(result["goal_success"])
        self.assertIn("unsupported changed columns", result["invalid_reasons"][0])

    def test_noncontiguous_transaction_and_bad_event_order_are_invalid(self):
        for mutation in ("transaction", "order"):
            events = deepcopy(self.events)
            if mutation == "transaction":
                events[-1]["transaction_id"] = events[0]["transaction_id"]
            else:
                events[-1]["seq"] = 1
            self.assertTrue(self.audit(events=events)["invalid_reasons"])

    def test_initial_target_collision_is_not_successful_creation(self):
        self.initial = deepcopy(self.final)
        result = self.audit(events=[])
        self.assertFalse(result["goal_success"])
        self.assertIn("already exists", " ".join(result["invalid_reasons"]))

    def test_absent_before_is_not_treated_as_explicit_null(self):
        events = deepcopy(self.events)
        del events[0]["before"]
        result = self.audit(events=events)
        self.assertFalse(result["goal_success"])
        self.assertIn("missing required event fields", " ".join(result["invalid_reasons"]))


if __name__ == "__main__":
    unittest.main()
