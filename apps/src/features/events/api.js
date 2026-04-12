import { apiRequest } from "../../lib/apiClient";

export async function fetchEvents(apiBase, { residentId = 1, limit = 500, signal } = {}) {
  return await apiRequest(apiBase, `/api/events?resident_id=${residentId}&limit=${limit}`, {
    signal,
  });
}

export async function fetchEventsSummary(apiBase, { residentId = 1, signal } = {}) {
  return await apiRequest(apiBase, `/api/events/summary?resident_id=${residentId}`, {
    signal,
  });
}

export async function updateEventStatus(apiBase, eventId, status) {
  return await apiRequest(apiBase, `/api/events/${eventId}/status`, {
    method: "PUT",
    body: { status },
  });
}

export async function fetchEventSkeletonClip(apiBase, eventId, { residentId = 1, signal } = {}) {
  return await apiRequest(
    apiBase,
    `/api/events/${eventId}/skeleton_clip?resident_id=${encodeURIComponent(residentId)}`,
    { signal }
  );
}
