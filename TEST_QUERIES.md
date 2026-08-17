# Best Test Queries for 3GPP RAG System

## Architecture & Registration (TS 23.501)
1. **"Explain the 5G core network architecture and the role of each NF"**
   - Tests: System knowledge, multi-section retrieval, architecture overview
   - Expected: AMF, SMF, UPF, PCF, AUSF, UDM roles

2. **"What is the difference between 5G-RAN and Core Network in terms of services?"**
   - Tests: Cross-spec knowledge, service-based architecture
   - Expected: TS 23.501 sections on service communication

3. **"How does UE registration happen in 5G?"**
   - Tests: Registration procedures, detailed process
   - Expected: Multiple steps, security, AMF involvement

## Procedures & PDU Sessions (TS 23.502)
4. **"Explain PDU session establishment procedure in detail"**
   - Tests: Complex multi-step procedure, citations
   - Expected: SMF role, UPF establishment, policy enforcement

5. **"What are the steps for UE-initiated service request?"**
   - Tests: Service procedure, state transitions
   - Expected: Clear procedural steps with section references

6. **"How does handover work between 5G cells?"**
   - Tests: Mobility procedures
   - Expected: Handover types, target selection

7. **"Explain Network Slicing and NSSAI usage"**
   - Tests: Cross-cutting concern, multiple sections
   - Expected: Slice selection, NSSAI computation

## Service-Based Architecture (TS 23.503)
8. **"What is Service-Based Architecture (SBA) in 5G?"**
   - Tests: SBA introduction, service concepts
   - Expected: Service descriptions, interfaces

9. **"How do NFs communicate using HTTP/2 in SBA?"**
   - Tests: Service communication details
   - Expected: HTTP/2, service requests/responses

## Security (TS 33.501)
10. **"Explain 5G security architecture and key derivation"**
    - Tests: Security procedures, TS 33.501 coverage
    - Expected: Authentication, key hierarchy

11. **"What is the 5G Authentication and Key Agreement (AKA) procedure?"**
    - Tests: Complex security procedure
    - Expected: AUSF, UDM, challenge-response

12. **"How is subscriber data protected during authentication?"**
    - Tests: Security data handling, TS 33.501
    - Expected: Privacy, data protection

## Cross-Cutting & Advanced
13. **"Explain QoS handling in 5G core and how policies are applied"**
    - Tests: Cross-spec knowledge (23.502, policy)
    - Expected: QoS rules, PCF involvement

14. **"What happens during a UE detach procedure?"**
    - Tests: State cleanup, multiple NFs
    - Expected: AMF, SMF, cleanup steps

15. **"Explain the difference between 5G and 4G (EPS) core architectures"**
    - Tests: Comparative analysis, legacy interop
    - Expected: Architecture differences, migration

## For Demo/Video
**Top 3 showcase queries:**
1. "Explain PDU session establishment procedure in detail"
2. "What is the role of each NF in 5G core architecture?"
3. "Explain 5G Authentication and Key Agreement procedure"

These cover:
- ✅ Complex procedures with multiple citations
- ✅ Architecture overview
- ✅ Security domain (TS 33.501)
- ✅ Multi-section retrieval
- ✅ Professional, detailed answers
